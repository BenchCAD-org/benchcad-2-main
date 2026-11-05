"""Harbor's installed Codex agent on HTTP: its model provider is the case's meter with supports_websockets =
false, so Codex sends its Responses requests as HTTP streams the meter can read (Codex 0.166.0 tries the
Responses WebSocket first and ends the turn when the upgrade is refused). Everything else
is Harbor's own; the provider is added with codex's `-c` overrides, nothing is written to its config."""
import shlex

from harbor.agents.installed.codex import Codex

SELECT = 'model_provider="benchcad_meter"'
PROVIDER = ('model_providers.benchcad_meter={ name = "benchcad-meter", base_url = "http://meter:8080/openai/v1", '
            'wire_api = "responses", supports_websockets = false, env_key = "OPENAI_API_KEY" }')


class CodexHttp(Codex):
    def build_cli_flags(self) -> str:
        own = f"-c {shlex.quote(SELECT)} -c {shlex.quote(PROVIDER)}"
        return (super().build_cli_flags() + " " + own).strip()
