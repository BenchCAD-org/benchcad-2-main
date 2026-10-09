"""Which answer a record was scored on, and whether the file on disk is still that answer.

A results record names its answer by path (`step`: final.step, pred_graph.json, or an assembly task's
fixed submission directory). A path says where an answer was, not which one. When two lanes shared an
episode directory, the second lane's final.step replaced the first's. Every rescore of the first
lane's records then judged the second lane's answer: three rescores in a row, each one looking fine.
So the harness records the answer's digest when the episode ends:

    artifact_sha256     the digest
    artifact_sha256_of  "file": sha256 of the file's bytes (final.step, pred_graph.json)
                        "tree": sha256 over the files of the submission directory that the scorer reads
                                (parts/<id>.step, assembly/instances.json), "<relative path>\\0<file
                                sha256>\\0" for each, in sorted relative-path order

Anything that judges a record again first compares the answer on disk with that digest (`check`). A
record made before the field existed can be checked against a snapshot manifest ({"files": {absolute
path: sha256}}, as answers_snapshot_*.manifest.json lists them). With neither, the answer is unverified.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

from envs.common.submission import ASSEMBLY, INSTANCES, PARTS, STEP_SUFFIXES


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(files: dict[str, str]) -> str:
    """A directory's digest from its files' digests, keyed by relative POSIX path. The order is the
    paths' string order, so the same digest can be computed from a manifest without the files."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(files[rel].encode())
        h.update(b"\0")
    return h.hexdigest()


def scored_file(rel: str) -> bool:
    """Whether the scorer reads this file of a submission directory (envs/common/submission.py): a STEP
    file directly under parts/, and assembly/instances.json. Nothing else there is part of the answer --
    assembly.step is never scored -- and a .DS_Store Finder leaves, a __pycache__ or an editor's
    temporary file must not make an unchanged answer read as another one."""
    head, _, name = rel.partition("/")
    if head == PARTS and "/" not in name:
        return Path(name).suffix.lower() in STEP_SUFFIXES
    return rel == f"{ASSEMBLY}/{INSTANCES}"


def artifact_sha256(path) -> tuple[str, str] | None:
    """("file" | "tree", digest) of the answer at `path`; None when nothing is there."""
    p = Path(path)
    if p.is_dir():
        files = ((f.relative_to(p).as_posix(), f) for f in p.rglob("*") if f.is_file())
        return "tree", tree_sha256({rel: file_sha256(f) for rel, f in files if scored_file(rel)})
    if p.is_file():
        return "file", file_sha256(p)
    return None


class Snapshots:
    """Reference digests from snapshot manifests, for records made before artifact_sha256 existed.
    A path that two manifests list with different digests held two answers, and nothing can say
    which one a record was scored on; `reference` reports it as such."""

    def __init__(self, manifests=()):
        self._files: dict[str, tuple[str, str]] = {}           # path -> (sha256, manifest name)
        self._conflicts: dict[str, list[str]] = {}
        for m in manifests:
            name = Path(m).name
            for path, digest in json.loads(Path(m).read_text())["files"].items():
                seen = self._files.setdefault(path, (digest, name))
                if seen[0] != digest:
                    self._conflicts.setdefault(path, [f"{seen[1]} {seen[0][:12]}"]).append(f"{name} {digest[:12]}")
        self._paths = sorted(self._files)

    def reference(self, path: str) -> dict | None:
        """{"of", "sha256", "against"} for the answer the manifests list at `path`: a file entry, or
        the scored files under it as a directory. {"conflict": ...} when they disagree; None when
        unlisted."""
        path = str(path)
        if path in self._conflicts:
            return {"conflict": f"{path}: {', '.join(self._conflicts[path])}"}
        if path in self._files:
            digest, name = self._files[path]
            return {"of": "file", "sha256": digest, "against": name}
        prefix = path.rstrip("/") + "/"
        i = bisect.bisect_left(self._paths, prefix)
        under = []
        while i < len(self._paths) and self._paths[i].startswith(prefix):
            under.append(self._paths[i])
            i += 1
        if not under:
            return None
        under = [p for p in under if scored_file(p[len(prefix):])]
        clash = [p for p in under if p in self._conflicts]
        if clash:
            return {"conflict": "; ".join(f"{p}: {', '.join(self._conflicts[p])}" for p in clash)}
        return {"of": "tree", "sha256": tree_sha256({p[len(prefix):]: self._files[p][0] for p in under}),
                "against": ", ".join(sorted({self._files[p][1] for p in under}))}


def check(rec: dict, path, snapshots: Snapshots | None = None) -> dict:
    """Whether the answer at `path` (the record's `step`, mapped to this machine) is the one the
    record was scored on. `status` is:

      verified    it hashes to the record's artifact_sha256, or to a snapshot's digest of its answer
      mismatch    it does not (or two snapshots disagree on the answer): judging it again would score
                  another answer under this record's name
      missing     nothing is at `path`: judging it would score "no submission" over the record's score
      unverified  neither the record nor a snapshot says which answer it was
    """
    now = artifact_sha256(path)
    if now is None:
        return {"status": "missing", "why": f"nothing at {path}"}
    if rec.get("artifact_sha256"):
        ref = {"of": rec.get("artifact_sha256_of"), "sha256": rec["artifact_sha256"], "against": "record"}
    else:
        ref = None
        if snapshots is not None:
            ref = snapshots.reference(rec["step"])
            if ref is None and str(path) != rec["step"]:
                ref = snapshots.reference(str(path))
        if ref is None:
            return {"status": "unverified", "why": "no artifact_sha256 in the record and no snapshot lists its answer"}
        if "conflict" in ref:
            return {"status": "mismatch", "why": f"the snapshots disagree on the answer: {ref['conflict']}"}
    if now != (ref["of"], ref["sha256"]):
        return {"status": "mismatch", "why": f"{path} hashes to {now[0]} {now[1][:12]}, not {ref['of']} "
                f"{ref['sha256'][:12]} ({ref['against']})", "recorded": ref["sha256"], "on_disk": now[1],
                "against": ref["against"]}
    return {"status": "verified", "against": ref["against"]}
