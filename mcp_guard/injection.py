"""Prompt injection detection for MCP server metadata.

MCP clients forward server and capability metadata (names, descriptions and the
whole ``inputSchema``) to the model, so text hidden there is read as if it were
part of the prompt. This module extracts every such text field and matches it
against a pattern catalog.

The default catalog is stdlib-only. Strict mode additionally runs Little
Canary's ``StructuralFilter`` (optional ``little-canary`` dependency), a
deterministic regex and decode-then-recheck layer. Neither mode makes model,
API or network calls. Pattern matching is evidence, not proof: a match means
the text resembles a known injection technique.
"""

from __future__ import annotations

import base64
import json
import re
import unicodedata
from bisect import bisect_left
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .models import MCPCapability, MCPManifest, RiskLevel

if TYPE_CHECKING:
    from little_canary import StructuralFilter

# Bounds that keep scanning linear in the manifest size. Text beyond them is not
# inspected and is reported as "uninspected-metadata" instead of being skipped silently.
MAX_FIELD_CHARS = 65_536
MAX_SCHEMA_DEPTH = 64
# Some Little Canary patterns backtrack quadratically on long input, so strict mode
# feeds it overlapping windows (its own default input bound is 4000 characters).
# Any match of up to CANARY_WINDOW_OVERLAP characters falls entirely inside one window;
# longer matches that straddle a window boundary can be missed. Encoded tokens longer
# than the overlap are therefore extracted from the whole field and decoded here.
CANARY_WINDOW_CHARS = 2_000
CANARY_WINDOW_OVERLAP = 200

DETECTOR_BUILTIN = "mcp-guard"
DETECTOR_CANARY = "little-canary"


class InjectionEngineUnavailableError(ImportError):
    """Raised when strict mode is requested but Little Canary is not installed."""


@dataclass(frozen=True)
class MetadataField:
    """A model-visible text field extracted from a manifest."""

    location: str
    text: str
    is_identifier: bool = False
    uninspected_reason: str = ""


@dataclass(frozen=True)
class InjectionMatch:
    """A single detection in one metadata field."""

    location: str
    injection_type: str
    level: RiskLevel
    evidence: str
    detector: str
    quoted: bool = False


@dataclass(frozen=True)
class _Pattern:
    injection_type: str
    level: RiskLevel
    regex: re.Pattern[str]


def _p(injection_type: str, level: RiskLevel, pattern: str) -> _Pattern:
    return _Pattern(injection_type, level, re.compile(pattern, re.IGNORECASE | re.MULTILINE))


_H = RiskLevel.HIGH
_M = RiskLevel.MEDIUM

# HIGH: phrasing whose only plausible purpose is to redirect the model.
# MEDIUM: phrasing that also has legitimate uses and needs human review.
PATTERNS: tuple[_Pattern, ...] = (
    # Instruction override
    _p(
        "instruction-override",
        _H,
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:of\s+)?(?:the\s+|your\s+|my\s+)?"
        r"(?:previous|prior|above|earlier|preceding|former|original|system)\s+"
        r"(?:instructions?|prompts?|rules|directions|directives|guidelines|context)\b",
    ),
    _p(
        "instruction-override",
        _H,
        r"\b(?:disregard|ignore|forget|override|bypass)\s+(?:all\s+|any\s+)?(?:of\s+)?"
        r"(?:your|its|the\s+system)\s+"
        r"(?:instructions|guidelines|rules|system\s+prompt|programming|training)\b",
    ),
    _p(
        "instruction-override",
        _H,
        r"\bignore\s+(?:everything|anything|all)\s+(?:above|before|prior|previously)\b",
    ),
    _p(
        "instruction-override",
        _H,
        r"\b(?:new|updated|revised|real|actual|hidden|secret)\s+(?:system\s+)?instructions?\s*:",
    ),
    # Safety bypass
    _p(
        "safety-bypass",
        _H,
        r"\b(?:ignore|override|bypass|circumvent)\s+(?:(?:all|any|your|the|its)\s+)*"
        r"(?:safety|guardrails?|alignment|ethics|ethical\s+guidelines|moderation|safeguards?"
        r"|content\s+(?:polic(?:y|ies)|filters?)"
        r"|security\s+(?:checks?|polic(?:y|ies)|rules|restrictions|measures|controls))\b",
    ),
    # Role hijacking
    _p(
        "role-hijack",
        _H,
        r"\byou\s+are\s+now\s+(?:an?\s|the\s|in\s|my\s|dan\b|unrestricted|unfiltered"
        r"|jailbroken|free\b|acting\b|operating\b|playing\b)",
    ),
    _p(
        "role-hijack",
        _H,
        r"\byou\s+are\s+no\s+longer\s+(?:an?\s|bound|restricted|limited|required)",
    ),
    _p("role-hijack", _H, r"\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must|should)\b"),
    _p(
        "role-hijack",
        _H,
        r"\byour\s+(?:new|real|true|actual)\s+(?:role|task|goal|purpose|instructions|objective)"
        r"\s+(?:is|are)\b",
    ),
    _p(
        "role-hijack",
        _H,
        r"\b(?:act|behave|respond)\s+as\s+(?:if\s+you\s+(?:are|were)\s+)?(?:an?\s+)?"
        r"(?:unrestricted|unfiltered|uncensored|jailbroken|evil)\b",
    ),
    _p("role-hijack", _M, r"\bpretend\s+(?:to\s+be|(?:that\s+)?you\s+are)\b"),
    # Jailbreak personas and modes
    _p(
        "jailbreak",
        _H,
        r"\bDAN\s+mode\b|\bdo\s+anything\s+now\b"
        r"|\b(?:god|jailbreak|jailbroken|unrestricted|unfiltered|uncensored)\s+mode\b",
    ),
    _p("jailbreak", _M, r"\bjailbr(?:eak|eaks|oken|eaking)\b"),
    # Fake authority: chat-template tokens and role turns
    _p(
        "fake-authority",
        _H,
        r"<\|(?:im_start|im_end|system|endoftext|eot_id|start_header_id|end_header_id)\|>"
        r"|\[/?INST\]|<<\s*/?SYS\s*>>",
    ),
    _p(
        "fake-authority",
        _H,
        r"\b(?:system|admin|administrator|developer|root)\s+"
        r"(?:override|instructions?|directive|command)\s*:",
    ),
    _p("fake-authority", _M, r"\[\s*(?:system|admin|developer)\s*(?:message|prompt|note)?\s*\]"),
    _p("fake-authority", _M, r"^[ \t]*(?:system|assistant|developer)[ \t]*:[ \t]*\S"),
    _p(
        "fake-delimiter",
        _M,
        r"^[ \t]*(?:#{2,}|-{3,}|={3,})[ \t]*"
        r"(?:system\b|end\s+of\s+(?:instructions|prompt|context))",
    ),
    _p("directive-tag", _M, r"<\s*/?\s*(?:important|system|instructions?|secret|hidden)\s*>"),
    # Concealment from the user (tool poisoning)
    _p(
        "concealment",
        _H,
        r"\b(?:do\s+not|don'?t|never)\s+(?:tell|inform|reveal\s+(?:this|it)\s+to"
        r"|disclose\s+(?:this|it)\s+to)\b[^.\n]{0,80}?\buser\b(?!['\u2019]s)",
    ),
    _p(
        "concealment",
        _H,
        r"\b(?:hide|conceal)\s+(?:this|these|it)\s+from\s+the\s+user\b"
        r"|\bkeep\s+(?:this|these|it)\s+(?:secret|hidden)\s+from\s+the\s+user\b",
    ),
    _p("concealment", _M, r"\b(?:do\s+not|don'?t|never)\s+mention\s+(?:that|this|it)\b"),
    _p("concealment", _M, r"\bwithout\s+(?:telling|informing|alerting)\s+the\s+user\b"),
    # Prompt extraction
    _p(
        "prompt-extraction",
        _H,
        r"\b(?:reveal|print|output|repeat|show|leak|return|include|dump|disclose)\s+(?:me\s+)?"
        r"(?:your|the)\s+(?:full\s+|entire\s+|complete\s+)?"
        r"(?:system\s+prompt|initial\s+instructions|hidden\s+instructions|original\s+instructions)",
    ),
    # Exfiltration
    _p(
        "exfiltration",
        _M,
        r"\b(?:read|cat|open|load|send|include|pass|attach|upload|forward|copy)\b[^.\n]{0,60}?"
        r"(?:\.ssh/(?:id_\w+|authorized_keys|config)|\.aws/credentials|\.netrc\b"
        r"|\.git-credentials|\.docker/config\.json|\.kube/config|\.cursor/mcp\.json"
        r"|\bid_(?:rsa|ed25519|ecdsa)\b|/etc/(?:passwd|shadow)\b)",
    ),
    _p(
        "exfiltration",
        _M,
        r"\b(?:all|every|any)\s+(?:emails?|messages?)\s+(?:must|should|have\s+to)\s+(?:also\s+)?"
        r"(?:be\s+)?(?:sent|forwarded|copied|cc'?d|bcc'?d|redirected)\s+to\b",
    ),
)

# Little Canary 0.4 reason prefixes -> (injection type, level). Checked in order.
# Unknown reasons fall back to MEDIUM so a newer Little Canary never loses findings.
_CANARY_REASONS: tuple[tuple[str, str, RiskLevel], ...] = (
    ("Direct injection", "instruction-override", _H),
    ("Role hijacking", "role-hijack", _H),
    ("Injection:", "fake-authority", _H),
    ("Extraction attempt", "prompt-extraction", _M),
    ("Known jailbreak: DAN", "jailbreak", _H),
    ("Known jailbreak: hypothetical", "jailbreak", _H),
    ("Known jailbreak", "jailbreak", _M),
    ("Encoded payload", "encoded-payload", _H),
    ("Encoding:", "encoded-payload", _M),
    ("Code injection", "code-injection", _M),
    ("Boundary attack: fake special token", "fake-authority", _H),
    ("Boundary attack", "fake-delimiter", _M),
)
# Length and character checks are not injection evidence; the length bound and
# hidden-character detection are applied by this module instead.
_CANARY_IGNORED = (
    "Input exceeds maximum length",
    "Input contains control characters",
    "Input contains suspicious Unicode",
)

_LEVEL_ORDER = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}

_QUOTED_SPAN = re.compile(
    r"(?<!\w)(?:\"[^\"\n]{1,500}\"|'[^'\n]{1,500}'|`[^`\n]{1,500}`"
    r"|\u201c[^\u201d\n]{1,500}\u201d|\u2018[^\u2019\n]{1,500}\u2019)(?!\w)"
)
_SIMPLE_KEY = re.compile(r"[A-Za-z_$][A-Za-z0-9_$-]{0,63}")
_ZWJ = "\u200d"  # joins emoji sequences; removed before matching but not reported


def normalize(text: str) -> tuple[str, bool]:
    """Return (text prepared for matching, whether invisible characters were present).

    Unicode tag characters (U+E0020-U+E007E) are decoded to the ASCII they smuggle;
    other format and control characters (zero-width, bidi overrides, ANSI escapes)
    are removed so they cannot split a phrase. NFKC folds full-width lookalikes.
    """
    hidden = False
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xE0020 <= code <= 0xE007E:
            out.append(chr(code - 0xE0000))
            hidden = True
        elif ch in "\t\n\r":
            out.append(ch)
        elif unicodedata.category(ch) in ("Cf", "Cc"):
            hidden = hidden or ch != _ZWJ
        else:
            out.append(ch)
    return unicodedata.normalize("NFKC", "".join(out)), hidden


def _split_identifier(text: str) -> str:
    """Turn 'ignorePrevious_instructions' into 'ignore Previous instructions'."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.sub(r"[_\-.]+", " ", text)


def _evidence(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= 80 else collapsed[:77] + "..."


class _QuotedSpans:
    """Non-overlapping quoted spans of a text, for O(log n) containment checks."""

    def __init__(self, text: str) -> None:
        spans = [m.span() for m in _QUOTED_SPAN.finditer(text)]
        self._starts = [start for start, _ in spans]
        self._ends = [end for _, end in spans]

    def contains(self, start: int, end: int) -> bool:
        index = bisect_left(self._starts, start) - 1  # last span opening before start
        return index >= 0 and end <= self._ends[index]


# Cue words that make Little Canary 0.4 try ROT13 / reversed-text decoding of a window.
# Cues found anywhere in a field are carried into every window of that field.
_DECODE_CUE = re.compile(
    r"rot13|caesar|cipher|shift|decode this|decrypt|reverse|backward|sdrawkcab", re.IGNORECASE
)

# Encoded tokens Little Canary would decode, matched linearly over the whole field.
_BASE64_TOKEN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
_HEX_TOKEN = re.compile(r"(?:[0-9a-fA-F]{2}[ \t]*){10,}")


def _decode_long_tokens(text: str) -> Iterator[tuple[str, str]]:
    """Yield (encoding, decoded text) for encoded tokens too long for one window."""
    for found in _BASE64_TOKEN.finditer(text):
        token = found.group(0).rstrip("=")
        if len(token) <= CANARY_WINDOW_OVERLAP:
            continue
        if len(token) % 4 == 1:  # one dangling character cannot encode a byte
            token = token[:-1]
        decoded = base64.b64decode(token + "=" * (-len(token) % 4))
        yield "base64", decoded.decode("utf-8", errors="ignore")
    for found in _HEX_TOKEN.finditer(text):
        if found.end() - found.start() <= CANARY_WINDOW_OVERLAP:
            continue
        token = re.sub(r"[ \t]+", "", found.group(0))
        yield "hex", bytes.fromhex(token).decode("utf-8", errors="ignore")


def _key_segment(key: str) -> str:
    if _SIMPLE_KEY.fullmatch(key):
        return f".{key}"
    shown = key if len(key) <= 64 else key[:64] + "..."
    return f"[{json.dumps(shown)}]"


def _walk_schema(node: object, path: str, depth: int) -> Iterator[MetadataField]:
    if depth > MAX_SCHEMA_DEPTH:
        yield MetadataField(path, "", uninspected_reason=f"nested deeper than {MAX_SCHEMA_DEPTH}")
        return
    if isinstance(node, dict):
        for key, value in cast("dict[object, object]", node).items():
            key_text = str(key)
            child = path + _key_segment(key_text)
            yield MetadataField(child, key_text, is_identifier=True)
            yield from _walk_schema(value, child, depth + 1)
    elif isinstance(node, list):
        for index, item in enumerate(cast("list[object]", node)):
            yield from _walk_schema(item, f"{path}[{index}]", depth + 1)
    elif isinstance(node, str):
        yield MetadataField(path, node)


def capability_fields(capability: MCPCapability) -> Iterator[MetadataField]:
    """Yield every model-visible text field of a capability."""
    yield MetadataField("name", capability.name, is_identifier=True)
    yield MetadataField("description", capability.description)
    yield from _walk_schema(capability.input_schema, "inputSchema", 0)


def server_fields(manifest: MCPManifest) -> Iterator[MetadataField]:
    """Yield the server-level text fields of a manifest."""
    yield MetadataField("server.name", manifest.name, is_identifier=True)
    yield MetadataField("server.description", manifest.description)


class PromptInjectionDetector:
    """Match metadata fields against the injection catalog.

    With ``strict=True`` Little Canary's StructuralFilter also runs on every field.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self._canary: StructuralFilter | None = None
        if strict:
            try:
                from little_canary import StructuralFilter
            except ImportError as e:
                raise InjectionEngineUnavailableError(
                    "Strict injection scanning requires Little Canary. "
                    "Install the optional extra: pip install 'mcp-guard[canary]'"
                ) from e
            self._canary = StructuralFilter(max_input_length=CANARY_WINDOW_CHARS)

    def scan_field(self, field: MetadataField) -> list[InjectionMatch]:
        """Return the strongest match per injection type found in one field."""
        if field.uninspected_reason:
            return [self._uninspected(field.location, field.uninspected_reason)]
        if not field.text:
            return []

        best: dict[str, InjectionMatch] = {}

        def keep(match: InjectionMatch) -> None:
            current = best.get(match.injection_type)
            if current is None or _LEVEL_ORDER[match.level] > _LEVEL_ORDER[current.level]:
                best[match.injection_type] = match

        raw = field.text
        if len(raw) > MAX_FIELD_CHARS:
            keep(self._uninspected(field.location, f"longer than {MAX_FIELD_CHARS} characters"))
            raw = raw[:MAX_FIELD_CHARS]

        text, hidden = normalize(raw)
        if field.is_identifier:
            text = _split_identifier(text)
        if hidden:
            keep(
                InjectionMatch(
                    field.location,
                    "hidden-characters",
                    RiskLevel.MEDIUM,
                    "invisible format/control characters",
                    DETECTOR_BUILTIN,
                )
            )

        for match in self._builtin_matches(field.location, text):
            keep(match)

        if self._canary is not None:
            for reason in self._canary_reasons(self._canary, text):
                mapped = _map_canary_reason(reason)
                if mapped is not None:
                    keep(
                        InjectionMatch(
                            field.location, mapped[0], mapped[1], reason, DETECTOR_CANARY
                        )
                    )
            for encoding, decoded in _decode_long_tokens(text):
                found = self._decoded_injection(self._canary, normalize(decoded)[0])
                if found is not None:
                    keep(
                        InjectionMatch(
                            field.location,
                            "encoded-payload",
                            found[1],
                            f"{encoding} token decodes to {found[0]}",
                            found[2],
                        )
                    )

        return list(best.values())

    @staticmethod
    def _builtin_matches(location: str, text: str) -> Iterator[InjectionMatch]:
        """Yield the strongest catalog match per pattern, checking every occurrence."""
        spans: _QuotedSpans | None = None
        for pattern in PATTERNS:
            strongest: InjectionMatch | None = None
            for found in pattern.regex.finditer(text):
                spans = spans or _QuotedSpans(text)
                quoted = spans.contains(found.start(), found.end())
                level = RiskLevel.MEDIUM if quoted and pattern.level == _H else pattern.level
                if strongest is None or _LEVEL_ORDER[level] > _LEVEL_ORDER[strongest.level]:
                    strongest = InjectionMatch(
                        location,
                        pattern.injection_type,
                        level,
                        _evidence(found.group(0)),
                        DETECTOR_BUILTIN,
                        quoted,
                    )
                if not quoted:
                    break  # an unquoted match cannot be outranked
            if strongest is not None:
                yield strongest

    @staticmethod
    def _canary_reasons(canary: StructuralFilter, text: str) -> Iterator[str]:
        cues = sorted({m.group(0).lower() for m in _DECODE_CUE.finditer(text)})
        for window in _windows(text):
            missing = [cue for cue in cues if cue not in window.lower()]
            prefix = " ".join(missing) + ". " if missing else ""
            yield from canary.check(prefix + window).reasons

    def _decoded_injection(
        self, canary: StructuralFilter, decoded: str
    ) -> tuple[str, RiskLevel, str] | None:
        """Strongest match and its actual detector in decoded text."""
        candidates = [
            (m.injection_type, m.level, m.detector) for m in self._builtin_matches("", decoded)
        ]
        for reason in self._canary_reasons(canary, decoded):
            mapped = _map_canary_reason(reason)
            if mapped is not None:
                candidates.append((*mapped, DETECTOR_CANARY))
        return max(
            candidates,
            key=lambda c: (_LEVEL_ORDER[c[1]], c[2] == DETECTOR_CANARY),
            default=None,
        )

    @staticmethod
    def _uninspected(location: str, reason: str) -> InjectionMatch:
        return InjectionMatch(
            location, "uninspected-metadata", RiskLevel.LOW, reason, DETECTOR_BUILTIN
        )


def _windows(text: str) -> Iterator[str]:
    """Split text into overlapping windows that together cover all of it."""
    start = 0
    while True:
        yield text[start : start + CANARY_WINDOW_CHARS]
        if start + CANARY_WINDOW_CHARS >= len(text):
            return
        start += CANARY_WINDOW_CHARS - CANARY_WINDOW_OVERLAP


def _map_canary_reason(reason: str) -> tuple[str, RiskLevel] | None:
    if reason.startswith(_CANARY_IGNORED):
        return None
    for prefix, injection_type, level in _CANARY_REASONS:
        if reason.startswith(prefix):
            return injection_type, level
    return "other", RiskLevel.MEDIUM
