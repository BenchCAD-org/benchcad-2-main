"""Restore TopoDS_*.HashCode inside the sandbox image.

cadquery 2.3 calls it; cadquery-ocp 7.9 removed it. Without it every
`.faces()` / `.edges()` selector raises AttributeError, which is most of
CadQuery. The harness used to prepend this patch to each program it executed:
that fixed our own runs and left the image broken for anything else, so a plain
`python3 -c` in this container could not use a selector at all. It belongs to
the environment, not to one caller.

Applied at interpreter start. OCP imports in ~0.3s and only when something asks
for it, so the cost is paid by programs that were going to import it anyway.
"""

try:
    from OCP.TopoDS import (  # noqa: F401
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
                 TopoDS_Wire, TopoDS_Shell, TopoDS_Solid, TopoDS_Compound,
                 TopoDS_CompSolid):
        if not hasattr(_cls, "HashCode"):
            _cls.HashCode = lambda self, ub=2147483647: id(self) % ub

    def show_object(*_a, **_k):
        """Defined only inside CQ-editor; model code that calls it should not die."""

    import builtins

    if not hasattr(builtins, "show_object"):
        builtins.show_object = show_object
except Exception:  # noqa: BLE001 - a sandbox that cannot patch must still start
    pass

# OCC's STEP reader/writer narrate every transfer to stdout in green ANSI
# ("Statistics on Transfer (Write)", "Step File Name : ... Write Done", ...).
# The harness returns only the tail of a program's stdout to the model, and a
# model prints its measurements last, so on a round that also exports a STEP
# the banner pushes the model's own numbers out of the observation. Keep
# failures, drop the rest. Same block as the harness's SITECUSTOMIZE_PY.
try:
    from OCP.Message import Message, Message_Gravity
    for _p in Message.DefaultMessenger_s().Printers():
        _p.SetTraceLevel(Message_Gravity.Message_Fail)
except Exception:
    pass
