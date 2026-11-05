"""The verifier could not finish: no score (error 1), to be verified again."""
import json
import sys
from pathlib import Path

V = Path("/logs/verifier")
V.mkdir(parents=True, exist_ok=True)
(V / "reward.json").write_text(json.dumps({"reward": 0.0, "error": 1, "no_answer": 0}) + "\n")
(V / "detail.json").write_text(json.dumps({"status": "pending", "why": sys.argv[1] if len(sys.argv) > 1 else "unknown"}, indent=1) + "\n")
(V / "reward.txt").write_text("0\n")
