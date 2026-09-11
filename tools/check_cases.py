#!/usr/bin/env python
"""Validate cases against the case format (docs/CASE_FORMAT.md).

    python tools/check_cases.py envs/t2_realparts2assembly/cases          # every case in a tree
    python tools/check_cases.py examples                                  # the dev samples
    python tools/check_cases.py tests/fixtures/t2/case1 tests/fixtures/t5/case1         # specific cases
    python tools/check_cases.py --deep envs/t2_realparts2assembly/cases   # also rebuild each assembly
                                                                          # from parts+instances and
                                                                          # require IoU >= 0.999 vs gt.step

Exit status is the number of failing cases (0 = all clean). Legacy-layout
directories (no case.json) are reported as such, not as failures, unless
--strict is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from envs.common.caseformat import check_case, is_new_format  # noqa: E402


def _cases(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "case.json").exists() or (p / "gt").is_dir():
            out.append(p)
        elif p.is_dir():
            for d in sorted(p.iterdir()):
                if d.is_dir() and not d.name.startswith((".", "_")):
                    if (d / "case.json").exists() or (d / "gt").is_dir():
                        out.append(d)
                    else:                      # family/NN two-level layout
                        out += [x for x in sorted(d.iterdir()) if x.is_dir() and (x / "gt").is_dir()]
    if not out:
        # Deeper than two levels: examples/ is task<N>/cases/<id>, so the fixed
        # walk above finds nothing there and the tool reported "no cases found"
        # on a tree full of cases. Same rule as harness/run.py discover().
        out = [c.parent for p_ in map(Path, paths) if p_.is_dir()
               for c in sorted(p_.rglob("case.json"))]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--deep", action="store_true", help="rebuild assemblies and compare voxel IoU")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--strict", action="store_true", help="legacy layout counts as a failure")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    cases = _cases(a.paths)
    if not cases:
        print("no cases found", file=sys.stderr)
        return 1
    failed = legacy = 0
    for c in cases:
        if not is_new_format(c):
            legacy += 1
            if a.strict:
                failed += 1
            if not a.quiet:
                print(f"LEGACY  {c}")
            continue
        rep = check_case(c, deep=a.deep, res=a.res)
        if rep.ok:
            if not a.quiet:
                print(f"ok      {c}" + (f"   ({'; '.join(rep.warnings)})" if rep.warnings else ""))
        else:
            failed += 1
            print(f"FAIL    {c}")
            for e in rep.errors:
                print(f"          - {e}")
            for w in rep.warnings:
                print(f"          ? {w}")
    print(f"{len(cases)} cases: {len(cases) - failed - legacy} ok, {failed} failed, {legacy} legacy")
    return failed


if __name__ == "__main__":
    sys.exit(main())
