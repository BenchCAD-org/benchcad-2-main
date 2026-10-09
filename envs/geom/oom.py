"""What counts as running out of memory while scoring.

Out of memory is the machine's, never the answer's: it must not become a
score. Python raises MemoryError (numpy's _ArrayMemoryError is one), but
OCCT raises Standard_OutOfMemory, which OCP binds as a plain Exception. A
guard that names MemoryError alone lets the OCCT one fall into the broad
handler behind it -- score.py's legacy iou scored the submission 0.0 on it.
Every guard in the scorer catches this one tuple instead
(tests/test_memoryerror_is_infra.py checks that it does).
"""


def _occt() -> tuple:
    try:
        from OCP.Standard import Standard_OutOfMemory
    except ImportError:                                 # an env without OCP has no OCCT to run out
        return ()
    return (Standard_OutOfMemory,)


OOM_ERRORS: tuple = (MemoryError,) + _occt()
