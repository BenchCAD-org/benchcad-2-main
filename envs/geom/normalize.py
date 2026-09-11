"""Normalise to [0,1]^3 by the longest axis -- so a global translation or scaling does
not affect the score."""
from __future__ import annotations

from pathlib import Path

from .tessellate import tessellate_all

def normalized_mesh(step_path: Path):
    import numpy as np
    import trimesh
    verts, tris = tessellate_all(step_path)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    longest = (hi - lo).max()
    if longest < 1e-9:
        raise ValueError("degenerate geometry")
    verts = (verts - (lo + hi) / 2.0) / longest + 0.5
    return trimesh.Trimesh(vertices=verts, faces=tris, process=False)


