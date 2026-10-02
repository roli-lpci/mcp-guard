# MCP Guard

**Security scanner for MCP servers** — audit capabilities, detect risks, generate security reports.

![CI](https://github.com/yunaremaia/mcp-guard/actions/workflows/ci.yml/badge.svg) ![Docker](https://github.com/yunaremaia/mcp-guard/actions/workflows/docker.yml/badge.svg) ![License](https://img.shields.io/github/license/yunaremaia/mcp-guard) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Release](https://img.shields.io/github/v/release/yunaremaia/mcp-guard)

## Why

The MCP ecosystem exploded (67,000+ servers in 18 months), but security hasn't kept up:
- 87% of MCP servers fail high-trust thresholds
- 72% expose sensitive capabilities unnecessarily
- 53% rely on static API keys

**MCP Guard** helps you audit MCP servers before connecting agents to them.

## Install

`mcp-guard` is not published on PyPI. Install it directly from the repository:

```bash
pip install git+https://github.com/yunaremaia/mcp-guard.git
```

Requires Python 3.10 or newer.

Alternatively, run it without installing via Docker:

```bash
docker run --rm -v "$PWD:/work" ghcr.io/yunaremaia/mcp-guard:latest scan /work/my-mcp-server
```

## Quick Start

```bash
# Scan an MCP server directory
mcp-guard scan ./my-mcp-server

# Scan a specific config file
mcp-guard scan ./mcp.json

# Output as JSON
mcp-guard scan ./my-mcp-server --format json

# Output as SARIF (for GitHub Code Scanning)
mcp-guard scan ./my-mcp-server --format sarif --output results.sarif

# Fail CI if HIGH or CRITICAL findings
mcp-guard scan ./my-mcp-server --fail-on high

# Enforce security deny rules from a YAML config
mcp-guard scan ./my-mcp-server --config policy.yaml --deny

# Deny specific servers or tools on the fly
mcp-guard scan ./my-mcp-server --deny-server "untrusted-*" --deny

# Add Little Canary's structural filter to prompt injection checks
# (requires: pip install "mcp-guard[canary] @ git+https://github.com/yunaremaia/mcp-guard.git")
mcp-guard scan ./my-mcp-server --strict-injection

# Show server info without scanning
mcp-guard info ./my-mcp-server
```

## Deny Rules & Policy Enforcement

You can configure deny rules in a YAML file (e.g. `policy.yaml` or default `mcp-guard.yaml`). Deny rules support exact matches, wildcards (`*`), and tool-level scoping:

```yaml
# Security Policy Configuration
deny:
  # Block unverified or untrusted servers
  servers:
    - "malicious-server"
    - "github-*"
  # Block high-risk tools (exact name, wildcard, or scoped to server)
  tools:
    - "github/delete_repo"
    - "filesystem/write"
    - "sys_*"
```

Use the `--deny` flag to fail with exit code 1 whenever any denied server or tool is detected:

```bash
mcp-guard scan ./my-mcp-server --config policy.yaml --deny
```

## What It Detects

| Rule | Level | Description |
|------|-------|-------------|
| MCP001 | HIGH | Write operation without authentication |
| MCP002 | CRITICAL | Destructive operation without authentication |
| MCP003 | MEDIUM | Capability with excessive permissions |
| MCP004 | LOW | Capability without description |
| MCP005 | MEDIUM | Write capability without corresponding read |
| MCP006 | HIGH | Destructive operation without confirmation |
| MCP007 | MEDIUM / HIGH / CRITICAL | Explicitly disabled authentication ('auth': false) |
| MCP008 | LOW / MEDIUM / HIGH | Possible prompt injection in server or capability metadata |
| DENY001 | CRITICAL | Server matches security policy deny rule |
| DENY002 | CRITICAL | Tool capability matches security policy deny rule |

### Authentication Detection Semantics

`mcp-guard` validates that capability authentication fields contain truthy configuration rather than mere key presence:

- **Auth Required**: Detected when `auth`, `authorization`, or `security` is present with a truthy value (`true`, configuration object/dict, non-empty string, or non-empty list).
- **Auth Explicitly Disabled**: Flagged when `auth` or `authorization` is set to `false` or `"disabled"`. Capabilities explicitly disabling authentication trigger rule `MCP007` and cannot bypass `MCP001` (write) or `MCP002` (destructive) checks.
- **No Auth Field / Falsy**: Falsy values like `null`, `""`, `0`, or `{}` are treated as unauthenticated.
- **SARIF Integration**: SARIF 2.1.0 output records `properties.auth_status` as `"required"`, `"disabled"`, or `"unknown"` for each capability finding.


### Prompt Injection Detection (MCP008)

MCP clients pass server and tool metadata to the model, so instructions hidden there act like
part of the prompt. `MCP008` checks the server name and description, every capability name and
description, and every key and string value in `inputSchema` (parameter names, descriptions,
titles, `enum`/`default` values, nested objects, array `items`, `anyOf`/`$defs`). `$ref` is not
resolved; local definitions are scanned where they appear.

- **HIGH**: direct injection phrasing, e.g. "ignore previous instructions", "override safety
  checks", "you are now ...", "DAN mode", chat-template tokens such as `<|im_start|>`,
  "do not tell the user".
- **MEDIUM**: suspicious phrasing with legitimate uses, e.g. a bare "jailbreak", `<IMPORTANT>`
  tags, a line starting with `system:`, directions to read files such as `~/.ssh/id_rsa`, and
  invisible Unicode (zero-width, bidi, tag characters). A HIGH pattern inside quotes (for
  example a security tool describing an attack) is reported as MEDIUM with `quoted: "true"`,
  never suppressed.
- **LOW**: a field longer than 65,536 characters or a schema nested deeper than 64 levels;
  the remainder is not scanned.

Text is normalized before matching: invisible characters are removed, Unicode tag characters
are decoded to the ASCII they hide, and full-width lookalikes are folded (NFKC).

`--strict-injection` also runs [Little Canary](https://github.com/hermes-labs-ai/little-canary)'s
`StructuralFilter`. It adds a broader signature set and decodes base64/hex/ROT13 payloads
before re-checking them. It is deterministic pattern matching. It makes no model, API or
network calls and needs the optional `canary` extra. Strict mode finds more and has more false
positives (for example `sudo` or "developer mode" in a shell tool's description).

To keep scan time linear, the filter sees each field in overlapping 2,000-character windows
(200-character overlap). A Little Canary match longer than 200 characters that crosses a window
boundary can be missed. Base64 and hex tokens longer than 200 characters are decoded whole by
mcp-guard instead, and the decoded text is checked by both detectors. A long token nested
inside a decoded token is not decoded again. Cue words that enable ROT13 and reversed-text
decoding ("rot13", "reverse", ...) are passed to every window of the field. ROT13 or reversed
text longer than 200 characters can still be split across windows.

These are pattern matches, not proof of intent, and they do not catch rephrased or novel
injections. On the hand-written benign fixture in `tests/fixtures/benign_tools.json` (121
tools), the default mode flags 5 tools (4.13%, all MEDIUM) and strict mode flags 6 (4.96%).
Those rates describe that fixture only.

In JSON and SARIF output each MCP008 finding has `injection_type` (e.g. `instruction-override`,
`safety-bypass`, `role-hijack`, `jailbreak`, `fake-authority`, `concealment`,
`prompt-extraction`, `exfiltration`, `hidden-characters`, `encoded-payload`),
`metadata_field` (e.g. `inputSchema.properties.path.description`), `detector`
(`mcp-guard` or `little-canary`), `evidence` and `quoted` properties.

## Example Output

```
MCP Server
test-server v1.0.0
A test MCP server

Risk Score: CRITICAL

Summary
Total Capabilities    3
Critical              1
High                  1
Medium                0
Low                   0

Findings
Rule      Level       Capability          Message                              Suggestion
MCP002    CRITICAL    delete_database     Destructive operation without auth    Add authentication
MCP001    HIGH        create_user         Write operation without auth         Add authentication
```

## CI Integration

### GitHub Action

```yaml
name: MCP Security Scan
on: [push, pull_request]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install git+https://github.com/yunaremaia/mcp-guard.git
      - run: mcp-guard scan ./mcp-server --format sarif --output results.sarif
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

If this tool is useful to you, a star helps other people find it.

## Related tools

- **[driftcheck](https://github.com/yunaremaia/driftcheck)** — detect version drift between docs and toolchain files
- **[context-bridge](https://github.com/yunaremaia/context-bridge)** — persistent session memory for AI agents
- **[mcp-reconcile](https://github.com/yunaremaia/mcp-reconcile)** — reconcile conflicting MCP server definitions
- **[tool-call-retry](https://github.com/yunaremaia/tool-call-retry)** — retry failed tool calls with backoff

Part of a family of focused, single-purpose developer tools — each one does one thing
and does it well.

## License

MIT

## Sponsoring / Treasury

This project is MIT licensed and free to use. If you want to support its
maintenance, you can sponsor on GitHub or contribute to the development
treasury wallet on Solana:

- GitHub Sponsors: https://github.com/sponsors/yunaremaia
- Solana: [`Eeztv1nCYUt1fwGWpzKC948gaWfjejYCAuLtUMgzDWbW`](https://solana.com/solana-wallet?base=SOL&address=Eeztv1nCYUt1fwGWpzKC948gaWfjejYCAuLtUMgzDWbW)

## Trust boundary

Treat MCP tool responses as untrusted input until validated by your agent policy.
