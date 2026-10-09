"""subdivide_voxelized must give trimesh's voxelize_subdivide grid exactly,
whatever the grouping of faces (envs/geom/voxel.py)."""
from pathlib import Path

import numpy as np
import pytest
import trimesh
from trimesh.voxel import creation

from envs.geom.voxel import subdivide_voxelized


def _same(a, b):
    assert np.array_equal(np.asarray(a.transform), np.asarray(b.transform))
    assert np.array_equal(a.matrix, b.matrix)
    assert np.array_equal(a.sparse_indices, b.sparse_indices)          # same order too
    assert np.array_equal(a.copy().fill().matrix, b.copy().fill().matrix)


def _unit(m):
    m = m.copy()
    m.apply_translation(-m.bounds[0])
    m.apply_scale(1.0 / m.extents.max())
    return m


MESHES = {
    # long slivers down the barrel, the shape that blew up case175
    "cylinder": lambda: _unit(trimesh.creation.cylinder(radius=0.3, height=4.0, sections=48)),
    "annulus": lambda: _unit(trimesh.creation.annulus(r_min=0.2, r_max=0.5, height=2.0, sections=64)),
    "sphere": lambda: _unit(trimesh.creation.icosphere(subdivisions=3)),
    "box": lambda: _unit(trimesh.creation.box(extents=(3.0, 1.0, 0.5))),
}


@pytest.mark.parametrize("name", sorted(MESHES))
@pytest.mark.parametrize("res", [32, 128])
@pytest.mark.parametrize("budget", [1, 1000, 50_000])
def test_matches_trimesh(name, res, budget):
    m = MESHES[name]()
    ref = creation.voxelize_subdivide(m, pitch=1.0 / res)
    _same(subdivide_voxelized(m, 1.0 / res, budget=budget), ref)


def test_unreferenced_vertex_is_kept():
    m = MESHES["box"]()
    v = np.vstack([m.vertices, [[0.5, 2.0, 0.5]]])                   # far outside, no face
    m2 = trimesh.Trimesh(vertices=v, faces=m.faces, process=False)
    _same(subdivide_voxelized(m2, 1.0 / 32, budget=1), creation.voxelize_subdivide(m2, pitch=1.0 / 32))


def test_shuffled_faces_and_shared_vertices():
    m = MESHES["cylinder"]()
    rng = np.random.default_rng(0)
    m2 = trimesh.Trimesh(vertices=m.vertices, faces=m.faces[rng.permutation(len(m.faces))], process=False)
    _same(subdivide_voxelized(m2, 1.0 / 128, budget=7), creation.voxelize_subdivide(m2, pitch=1.0 / 128))


def test_max_iter_raises_like_trimesh():
    m = MESHES["cylinder"]()
    with pytest.raises(ValueError, match="max_iter exceeded"):
        creation.voxelize_subdivide(m, pitch=1.0 / 128, max_iter=2)
    with pytest.raises(ValueError, match="max_iter exceeded"):
        subdivide_voxelized(m, 1.0 / 128, max_iter=2, budget=1)


def test_trimesh_is_5x():
    # 4.x drops unreferenced vertices from subdivide_to_size's output, so the
    # grouped path would add cells the old code never produced (pyproject pins 5.x)
    assert int(trimesh.__version__.split(".")[0]) == 5


def _probe(m, res=128, budget=1):
    _same(subdivide_voxelized(m, 1.0 / res, budget=budget), creation.voxelize_subdivide(m, pitch=1.0 / res))


def test_degenerate_and_repeated_index_faces():
    m = MESHES["cylinder"]()
    f = np.vstack([m.faces, [[0, 0, 1], [2, 2, 2], [0, 1, 0]]])                # zero-area faces
    v = np.vstack([m.vertices, m.vertices[:1] + 0.5 * (m.vertices[1:2] - m.vertices[:1])])
    f = np.vstack([f, [[0, 1, len(v) - 1]]])                                  # collinear
    _probe(trimesh.Trimesh(vertices=v, faces=f, process=False))


def test_unmerged_soup():
    m = MESHES["annulus"]()
    tri = m.vertices[m.faces].reshape(-1, 3)                                  # every face its own vertices
    _probe(trimesh.Trimesh(vertices=tri, faces=np.arange(len(tri)).reshape(-1, 3), process=False))


def test_half_cell_ties():
    res = 32
    m = MESHES["box"]()
    v = (np.round(m.vertices * res * 4) / 4 + 0.5) / res                      # vertices on half cells
    _probe(trimesh.Trimesh(vertices=v, faces=m.faces, process=False), res=res)


def test_far_coordinates():
    m = MESHES["cylinder"]()
    v = m.vertices + 2.0 ** 21 / 128                                          # 2**21 cells from the origin
    _probe(trimesh.Trimesh(vertices=v, faces=m.faces, process=False))


def test_call_sites_take_the_grouped_path_unchanged(monkeypatch):
    """solid_voxels, asm_v1.surface_indices and score_asm._vox, through the
    grouped path (budget 1) and through trimesh's own function."""
    import envs.geom.voxel as V
    from envs.common import asm_v1, score_asm

    m = MESHES["cylinder"]()
    verts, tris = np.asarray(m.vertices), np.asarray(m.faces)
    orig = V.subdivide_voxelized

    from envs.common import score
    step = Path(__file__).parent / "fixtures/t1/case1/gt/gt.step"
    seen = []

    def run():
        return (V.solid_voxels(m, 64).matrix, asm_v1.surface_indices(verts, tris, 64),
                score_asm._vox(verts, tris, 64), score.iou_step_vs_step(step, step, res=32))

    def grouped_fn(mesh, pitch, **kw):
        seen.append(pitch)
        return orig(mesh, pitch, budget=1, **kw)

    monkeypatch.setattr(V, "subdivide_voxelized", grouped_fn)
    grouped = run()
    assert len(seen) == 5                       # solid_voxels, surface_indices, _vox, iou (a and b)
    monkeypatch.setattr(V, "subdivide_voxelized", lambda mesh, pitch, max_iter=10, edge_factor=2.0, **_:
                        creation.voxelize_subdivide(mesh, pitch=pitch, max_iter=max_iter, edge_factor=edge_factor))
    plain = run()
    for a, b in zip(grouped, plain):
        assert np.array_equal(np.asarray(a), np.asarray(b))
