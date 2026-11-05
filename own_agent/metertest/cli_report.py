"""Per agent CLI, what the stub test showed (cli_tests.sh): python cli_report.py JOBS STUBLOGS agent ..."""
import json
import sys
from pathlib import Path

jobs, logs, agents = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
for a in agents:
    trials = [t for t in (jobs / f"cli-{a}").glob("*__*") if t.is_dir()]
    if not trials:
        print(f"{a}: NO TRIAL"); continue
    t = trials[0]
    r = json.loads((t / "result.json").read_text()) if (t / "result.json").exists() else {}
    exc = (r.get("exception_info") or {})
    m = t / "artifacts/meter"
    usage = json.loads((m / "usage.json").read_text()) if (m / "usage.json").exists() else None
    reqs = [json.loads(l) for l in (m / "requests.jsonl").read_text().splitlines()] if (m / "requests.jsonl").exists() else []
    stub = [json.loads(l) for l in (logs / a / "requests.jsonl").read_text().splitlines()] if (logs / a / "requests.jsonl").exists() else []
    ae = r.get("agent_execution") or {}
    from datetime import datetime
    secs = None
    if ae.get("started_at") and ae.get("finished_at"):
        secs = round((datetime.fromisoformat(ae["finished_at"].replace("Z", "+00:00")) - datetime.fromisoformat(ae["started_at"].replace("Z", "+00:00"))).total_seconds())
    served = [x for x in reqs if not x.get("refused") and not x.get("websocket")]
    print(f"{a}: agent {(r.get('agent_info') or {}).get('version')} | {secs} s | exception {exc.get('exception_type')}")
    if usage:
        print(f"   meter: forwarded {len(served)} (generation {usage['generation_requests']}), output_tokens {usage['output_tokens']},"
              f" usage_unread {usage['usage_unread']}, refused {usage['refused']}, websocket_refused {usage['websocket_refused']},"
              f" include_usage_added {usage['include_usage_added']}, budget_hit {usage['budget_hit']}")
    else:
        print("   meter: NO RECORD")
    for x in reqs[:6]:
        print("   meter>", {k: x.get(k) for k in ("path", "status", "stream", "usage_read", "output_tokens", "refused", "websocket")})
    for x in stub[:4]:
        print("   stub >", {k: x.get(k) for k in ("method", "path", "stream", "stream_options", "upgrade")}, "tools:", (x.get("tools") or [])[:8])
