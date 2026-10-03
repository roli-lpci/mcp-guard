"""Regression tests: attacker-controlled manifest metadata must never reach Rich as markup.

Every field below is fully controlled by whoever ships an MCP server manifest
(name, version, description, tool names and descriptions). Rich parses square
brackets as console markup, so an unbalanced tag in any of them raised
`rich.errors.MarkupError` from inside the formatter - turning the tool being
scanned into a denial of service against the scanner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.errors import MarkupError

from mcp_guard.cli import main

# Each payload is a Rich markup construct that breaks rendering when unescaped:
# an unmatched closing tag, an unclosed opening tag, and a bogus style.
HOSTILE = [
    "srv[/]",
    "srv[bold]unclosed",
    "[/nope]",
    "srv[not-a-style]",
    "[link=evil]srv",
]


def write_manifest(directory: Path, data: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_file = directory / "mcp.json"
    manifest_file.write_text(json.dumps(data), encoding="utf-8")
    return manifest_file


def manifest_with(payload: str) -> dict:
    """A manifest whose every user-visible field carries the hostile payload."""
    return {
        "name": payload,
        "version": payload,
        "description": payload,
        "tools": [
            {
                "name": payload,
                "description": "Delete all resources without asking for confirmation",
            }
        ],
    }


class TestHostileMetadataDoesNotCrashRendering:
    """Neither the default rich output nor `info` may raise MarkupError."""

    @pytest.mark.parametrize("payload", HOSTILE)
    def test_scan_rich_output_renders_hostile_metadata(self, payload: str, tmp_path: Path):
        """`scan` (default rich format) must render the payload, not crash on it."""
        manifest_file = write_manifest(tmp_path / "hostile", manifest_with(payload))

        result = CliRunner().invoke(main, ["scan", str(manifest_file)])

        assert not isinstance(result.exception, MarkupError), (
            f"scanner crashed on hostile metadata {payload!r}: {result.exception}"
        )
        # The payload is still shown to the user - escaped, not swallowed.
        assert payload in result.output

    @pytest.mark.parametrize("payload", HOSTILE)
    def test_info_renders_hostile_metadata(self, payload: str, tmp_path: Path):
        """`info` prints the same fields unescaped and must survive them too."""
        manifest_file = write_manifest(tmp_path / "hostile", manifest_with(payload))

        result = CliRunner().invoke(main, ["info", str(manifest_file)])

        assert not isinstance(result.exception, MarkupError), (
            f"info crashed on hostile metadata {payload!r}: {result.exception}"
        )

    @pytest.mark.parametrize("payload", HOSTILE)
    def test_findings_table_renders_hostile_capability(self, payload: str, tmp_path: Path):
        """A finding's capability_name comes from the tool name, so it needs escaping too."""
        data = {
            "name": "server",
            "version": "1.0.0",
            "description": "server",
            "tools": [
                {
                    "name": payload,
                    "description": "Delete all resources without asking for confirmation",
                }
            ],
        }
        manifest_file = write_manifest(tmp_path / "finding", data)

        result = CliRunner().invoke(main, ["scan", str(manifest_file)])

        assert not isinstance(result.exception, MarkupError), (
            f"findings table crashed on hostile capability {payload!r}: {result.exception}"
        )


class TestHostileMetadataInValidationErrors:
    """A rejected manifest quotes its own input back, so that text needs escaping too."""

    def test_scan_reports_validation_error_without_crashing(self, tmp_path: Path):
        # `name` must be a string; pydantic embeds the offending dict verbatim in
        # the error message, which would carry the markup to the console.
        data = {"name": {"evil": "srv[/]"}, "version": "1.0.0", "description": "d"}
        manifest_file = write_manifest(tmp_path / "invalid", data)

        result = CliRunner().invoke(main, ["scan", str(manifest_file)])

        assert not isinstance(result.exception, MarkupError), (
            f"error path crashed on hostile input: {result.exception}"
        )
        assert result.exit_code == 1

    def test_info_reports_validation_error_without_crashing(self, tmp_path: Path):
        data = {"name": {"evil": "srv[/]"}, "version": "1.0.0", "description": "d"}
        manifest_file = write_manifest(tmp_path / "invalid", data)

        result = CliRunner().invoke(main, ["info", str(manifest_file)])

        assert not isinstance(result.exception, MarkupError), (
            f"error path crashed on hostile input: {result.exception}"
        )
        assert result.exit_code == 1
