#!/usr/bin/env python
"""Score one submission against one case.

    python tools/verify_case.py <case dir> <your.step>          a part task (T1 / T3)
    python tools/verify_case.py <case dir> <dir>/submission     an assembly task (T2 / T4 / T5):
                                                                the submission directory
                                                                (docs/CASE_FORMAT.md, "Submission
                                                                layout"); a single STEP is the old
                                                                layout and still works

e.g. python tools/verify_case.py envs/t1_drawing2part/cases/PART-0161 my.step
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.common.score_case import fmt, score_case  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    case, step = Path(sys.argv[1]), Path(sys.argv[2])
    gt = case / "gt/gt.step"
    if not gt.exists():
        print(f"{gt} does not exist")
        return 2
    if not step.exists():
        print(f"{step} does not exist")
        return 2
    print(fmt(score_case(case, step)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
