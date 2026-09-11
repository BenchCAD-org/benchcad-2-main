"""One verifier per kind, and they never import one another.

Thin wrappers: the implementations stay where they are (envs/common/); this package only
provides the stable entry points that task.toml points at. That way not a line of the
assembly-side score_asm/rubric_asm logic has to change, while the runtime no longer needs
to sniff directories.
"""
