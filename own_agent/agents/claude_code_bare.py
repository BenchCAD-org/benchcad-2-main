"""Harbor's installed Claude Code agent with `--bare` added to its command line: no hooks, plugins, skills,
auto-memory or CLAUDE.md discovery, API-key auth only. Everything else (install, stream-json log, ATIF
trajectory, token and cost accounting) is Harbor's own.

    harbor run -p OUT --agent agents.claude_code_bare:ClaudeCodeBare -m <model> --ak reasoning_effort=<effort> ...
(own_agent/ on PYTHONPATH; benchcad.py sets it, and the meter as ANTHROPIC_BASE_URL)."""
from harbor.agents.installed.claude_code import ClaudeCode


class ClaudeCodeBare(ClaudeCode):
    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        return (flags + " --bare").strip()
