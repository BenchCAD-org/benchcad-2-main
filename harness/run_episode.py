#!/usr/bin/env python
"""Deprecated shim: the Anthropic-only runner is now `harness/run.py`.

    uv run python harness/run.py --model anthropic/claude-opus-5 --cases <case>

Kept so existing commands keep working, with the SAME defaults this script had
(claude-sonnet-5, 10 rounds) -- run.py's own defaults differ, and silently
changing the model or the round count under an existing command would change
what it costs and what it scores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as _run                                         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    model = a.model if "/" in a.model else "anthropic/" + a.model
    argv = ["--model", model, "--cases", a.case, "--rounds", str(a.rounds)]
    if a.work:
        argv += ["--work", a.work]
    print(f"[run_episode.py is a shim for run.py: {' '.join(argv)}]", file=sys.stderr)
    sys.argv = ["run.py", *argv]
    return _run.main()


if __name__ == "__main__":
    sys.exit(main())
