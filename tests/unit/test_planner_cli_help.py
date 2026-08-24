"""Unit tests for planner CLI help commands (no network needed)."""

import subprocess

import pytest


def run_cli(*args):
    """Run CLI command and return result."""
    cmd = ["planner"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


@pytest.mark.unit
class TestHelp:
    """Test help and usage commands."""

    def test_help(self):
        """Test help command."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "planner" in result.stdout.lower()

    def test_plan_help(self):
        """Test plan help."""
        result = run_cli("plan", "--help")
        assert result.returncode == 0
        assert "--model" in result.stdout
