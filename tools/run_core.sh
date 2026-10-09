#!/usr/bin/env bash
# One command from a clean clone to a BenchCAD 2.0 Core score.
#
#   tools/run_core.sh --model <provider>/<model-id> [--effort low,medium,high] [options]
#
#   --model     anthropic/claude-haiku-5-5, openai/gpt-6-astra, gemini/<id>, ... (required)
#   --effort    comma-separated levels, one full run each (default: the provider's ladder --
#               gemini low,medium,high; anthropic and openai low,medium,high,xhigh,max)
#   --rounds N  rounds per episode (default 30, the published setting; anything else is a smoke run)
#   --cases D   run a subset: a task or case directory inside the downloaded Core tree
#   --workers N episodes in flight per effort (default 4)
#   --data D    where Core is downloaded (default ~/.cache/benchcad/benchcad-2.0-core)
#   --dry-run   check everything, print the commands, run nothing
#   -- ...      anything after -- is passed to harness/run.py
#
# Keys come from the environment: HF_TOKEN (an account with access to the gated
# dataset) and the provider's key (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, ...).
# Re-running the same command resumes. Results: results/core/<model>/.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET_REPO=BenchCAD/benchcad-2.0-core
DATASET_NAME=benchcad-2.0-core
DATASET_VERSION=1.0

MODEL="" EFFORTS="" ROUNDS=30 CASES="" WORKERS=4 DRY=0 EXTRA=()
DATA="${BENCHCAD_DATA:-$HOME/.cache/benchcad/$DATASET_NAME}"
die() { echo "run_core: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --effort) EFFORTS=$2; shift 2 ;;
        --rounds) ROUNDS=$2; shift 2 ;;
        --cases) CASES=$2; shift 2 ;;
        --workers) WORKERS=$2; shift 2 ;;
        --data) DATA=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n 2,19p "$0"; exit 0 ;;
        *) die "unknown argument $1 (see --help)" ;;
    esac
done
[ -n "$MODEL" ] || die "--model <provider>/<model-id> is required (see --help)"
step() { echo "[run_core] $*"; }

# 1. tools: uv is installed if missing; Docker cannot be installed for you.
if ! command -v uv >/dev/null 2>&1; then
    step "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    msg="Docker is required and must be usable by this user (Ubuntu: curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker \$USER, then log in again)"
    [ "$DRY" = 1 ] && step "WARNING: $msg" || die "$msg"
fi

# 2. the environment
if [ "$DRY" = 0 ]; then step "uv sync"; uv sync --frozen --quiet; fi
PY=(uv run --no-sync python)
[ -z "${BENCHCAD_PYTHON:-}" ] || PY=("$BENCHCAD_PYTHON")

# 3. the sandbox image; the model's code never runs outside it
case "$(uname -m)" in arm64|aarch64) ARCH=arm64 ;; *) ARCH=amd64 ;; esac
export CADENV_DOCKER_IMAGE="${CADENV_DOCKER_IMAGE:-benchcad-sandbox:$ARCH}"
unset CADENV_LOCAL
if ! docker image inspect "$CADENV_DOCKER_IMAGE" >/dev/null 2>&1; then
    if [ "$DRY" = 1 ]; then step "would build $CADENV_DOCKER_IMAGE from sandbox/"
    else step "building $CADENV_DOCKER_IMAGE (once)"; docker build -q -t "$CADENV_DOCKER_IMAGE" sandbox/ >/dev/null; fi
fi

# 4. the dataset: downloaded once, checked every time
verify() {
    [ -f "$DATA/MANIFEST.sha256" ] || return 1
    if command -v sha256sum >/dev/null 2>&1; then (cd "$DATA" && sha256sum -c --quiet MANIFEST.sha256)
    else (cd "$DATA" && shasum -a 256 -c --quiet MANIFEST.sha256); fi
}
if ! verify >/dev/null 2>&1; then
    [ "$DRY" = 1 ] && die "no verified dataset at $DATA (a dry run does not download)"
    step "downloading $DATASET_REPO to $DATA"
    uvx --quiet --from huggingface_hub hf download "$DATASET_REPO" --repo-type dataset --local-dir "$DATA" >/dev/null \
        || die "download failed: $DATASET_REPO is gated -- request access on its Hugging Face page, then set HF_TOKEN (or run: uvx --from huggingface_hub hf auth login)"
    verify || die "$DATA does not match MANIFEST.sha256; delete it and run again"
fi
step "dataset verified: $DATA"

# 5. preflight, before any spend: dataset version, scorer seal, model, efforts, keys
EFFORTS=$("${PY[@]}" - "$MODEL" "$EFFORTS" "$DATA" "$DATASET_NAME" "$DATASET_VERSION" "${CASES:-$DATA}" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "harness"); sys.path.insert(0, ".")
import run as R
from envs.common.ecad_graph.spatial_reference import scorer_digest
model, efforts, data, name, version, cases = sys.argv[1:]
ds = json.loads((Path(data) / "dataset.json").read_text())
if (ds.get("name"), ds.get("version")) != (name, version):
    sys.exit(f"run_core: this harness expects {name} {version}, the download is {ds.get('name')} {ds.get('version')}; "
             f"git pull, or delete {data} and run again")
if ds.get("scorer_digest") != scorer_digest():
    sys.exit("run_core: the dataset is bound to a different T6 scorer than this checkout; git pull")
n = sum(1 for _ in Path(data).rglob("case.json"))
if n != ds.get("n_cases"):
    sys.exit(f"run_core: {n} cases on disk, dataset.json says {ds.get('n_cases')}")
prefix, prov, _ = R.split_model(model)
# Unset: the provider's ladder, every level but none (a provider without a knob runs once).
levels = [e for e in efforts.split(",") if e] or \
    [e for e in (R.PROVIDER_EFFORTS.get(prefix) or (None,)) if e != "none"]
for e in levels:
    R.check_effort(prefix, e)
if prov.kind == "gemini":
    print("gemini auth:", R.gemini_auth()[1], file=sys.stderr)
elif prov.kind != "mock":
    R.resolve_key(prefix, prov)
print(",".join("-" if e is None else e for e in levels))
EOF
) || exit 1

TAG=$(echo "$MODEL" | tr '/:' '__')
OUT="results/core/$TAG"
mkdir -p "$OUT"
[ "$ROUNDS" = 30 ] || step "SMOKE RUN: rounds=$ROUNDS, not a reportable score (reported numbers use 30)"

# 6. one run per effort, resumable
FILES=()
for E in ${EFFORTS//,/ }; do
    EFF=(); [ "$E" = "-" ] || EFF=(--effort "$E")
    NAME=${E/-/default}
    CMD=("${PY[@]}" -u harness/run.py --model "$MODEL" ${EFF[@]+"${EFF[@]}"} --cases "${CASES:-$DATA}"
         --rounds "$ROUNDS" --rep 0 --workers "$WORKERS" --resume --out "$OUT/$NAME.json"
         --work "work/core/$TAG/$NAME" ${EXTRA[@]+"${EXTRA[@]}"})
    FILES+=("$OUT/$NAME.json")
    if [ "$DRY" = 1 ]; then echo "[run_core] would run: ${CMD[*]}"; continue; fi
    step "effort $NAME: $OUT/$NAME.json (log $OUT/$NAME.log)"
    "${CMD[@]}" 2>&1 | tee -a "$OUT/$NAME.log" | grep -E '^\s*\[[0-9]+/[0-9]+\]|cases ->' || true
done

# 7. the score
[ "$DRY" = 1 ] && { step "dry run: all checks passed"; exit 0; }
"${PY[@]}" tools/core_score.py "${FILES[@]}" --dataset "${CASES:-$DATA}" --json "$OUT/core_summary.json"
