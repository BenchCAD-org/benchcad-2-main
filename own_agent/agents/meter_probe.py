"""A test agent for the meter (metertest/make_metertest.sh): it writes the reference answer (the test task
carries it in /probe_answer, which no real task does), then asks the model API for completions through the
meter until it is refused, and tries the network around it. What it saw goes to /logs/agent/probe.json."""
from harbor.agents.base import BaseAgent

PROBE = r'''
import json, socket, urllib.request, urllib.error
out = {"calls": [], "direct": {}}
def call(stream):
    body = json.dumps({"model": "stub", "max_tokens": 10, "stream": stream,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    req = urllib.request.Request("http://meter:8080/anthropic/v1/messages", data=body, method="POST",
                                 headers={"content-type": "application/json", "x-api-key": "stub",
                                          "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()[-300:]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]
for i in range(8):
    st, tail = call(stream=bool(i % 2))
    out["calls"].append({"i": i, "stream": bool(i % 2), "status": st, "tail": tail})
    if st == 400:
        break
for name, url in [("anthropic_direct", "https://api.anthropic.com/v1/messages"), ("example", "https://example.com/"),
                  ("pypi", "https://pypi.org/simple/")]:
    try:
        urllib.request.urlopen(url, timeout=10)
        out["direct"][name] = "REACHED"
    except urllib.error.HTTPError as e:
        out["direct"][name] = f"REACHED (HTTP {e.code})"
    except Exception as e:
        out["direct"][name] = f"blocked: {type(e).__name__}: {str(e)[:120]}"
with urllib.request.urlopen("http://meter:8080/_meter/status", timeout=10) as r:
    out["status"] = json.loads(r.read())
open("/logs/agent/probe.json", "w").write(json.dumps(out, indent=1))
print(json.dumps(out)[:2000])
'''


class MeterProbe(BaseAgent):
    @staticmethod
    def name() -> str:
        return "meter-probe"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment) -> None:
        return

    async def run(self, instruction, environment, context) -> None:
        await environment.exec(command="cp -r /probe_answer/. /work/", timeout_sec=60)
        r = await environment.exec(command="python3 - <<'PY'\n" + PROBE + "\nPY", timeout_sec=600)
        (self.logs_dir / "probe.stdout").write_text((r.stdout or "") + (r.stderr or ""))
