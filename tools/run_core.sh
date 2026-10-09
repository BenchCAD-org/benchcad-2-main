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
#   --workers N episodes in flight per effort (default: two per GB of available memory left
#               after run.py and the scorers, at most 2 x CPUs and 32; the value chosen is printed)
#   --data D    where Core is downloaded (default ~/.cache/benchcad/benchcad-2.0-core)
#   --dry-run   every free step (uv, the environment, the image, the scorer self-test,
#               dataset access), then print the commands; no download, no API call
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

MODEL="" EFFORTS="" ROUNDS=30 CASES="" WORKERS="" DRY=0 EXTRA=()
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

# The memory budget, from MemAvailable (BENCHCAD_NPROC / BENCHCAD_MEM_GB override what is
# measured, for tests). run.py 3 GB and 3 GB per scorer are reserved; the rest gives two
# episodes per GB (two per CPU at most, 32 at most) and one concurrent sandbox execution per
# 2 GB; scorers are max(1, min(4, GB / 5)). Sandbox executions and scorers together stay
# within the CPUs (each execution takes one; 16 on 8 cores starved the scorers), whatever
# --workers is. Another run_core on this machine halves it.
# Measured 2026-10-09 on an 8 vCPU / 16 GB box (14 GB available), Gemini flash at low:
#   5 workers  ~60 cases/h,  run.py peak RSS 2.8 GB, least available 8.1 GB, load 6.2
#   12 workers ~161 cases/h, run.py RSS 0.6 GB, least available 12.6 GB, load 4.4; per-case
#   wall time the same (median 58 vs 52 s); no OOM; at most 3 scorers and 3 containers
#   Haiku high, 30 rounds, 16 workers: run.py peak 2.4-2.8 GB
# so an episode costs well under the 1 GB the budget gave it; scorers measured 1-1.4 GB RSS
# (cap 4 GB), and T5/T6 scorers spike, so they keep 3 GB each. That box now gets 10 workers.
mem_budget() {
    local cpus mem s reserve free w e
    cpus=${BENCHCAD_NPROC:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}
    if [ -n "${BENCHCAD_MEM_GB:-}" ]; then mem=$BENCHCAD_MEM_GB
    elif [ -r /proc/meminfo ]; then mem=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    else mem=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 8589934592) / 2147483648 )); fi   # macOS: half of RAM
    [ "${SHARED:-0}" = 1 ] && mem=$(( mem / 2 ))
    s=$(( mem / 5 )); [ "$s" -gt 4 ] && s=4; [ "$s" -lt 1 ] && s=1
    reserve=$(( 3 + 3 * s ))
    free=$(( mem - reserve )); [ "$free" -lt 0 ] && free=0
    w=$(( 2 * free )); [ "$w" -gt $(( 2 * cpus )) ] && w=$(( 2 * cpus )); [ "$w" -gt 32 ] && w=32; [ "$w" -lt 1 ] && w=1
    e=$(( free / 2 )); [ "$e" -gt $(( cpus - s )) ] && e=$(( cpus - s )); [ "$e" -lt 1 ] && e=1   # execs + scorers <= CPUs
    echo "$w $s $e $cpus CPUs, $mem GB available: run.py 3 GB + $s scorers x 3 GB reserved, $free GB for $w episodes, $e sandbox executions at once"
}

# 1. tools: uv is installed if missing (and found again at ~/.local/bin on the next run);
#    Docker cannot be installed for you.
[ -x "$HOME/.local/bin/uv" ] && export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    step "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
    step "uv installed in ~/.local/bin; for your own shell: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
docker_ok() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }
if ! docker_ok; then
    # a freshly booted box may still be finishing its own setup: look again for a minute
    for _ in 1 2 3 4 5 6; do [ -n "${BENCHCAD_SKIP_SETUP:-}" ] && break; sleep 10; docker_ok && break; done
fi
if ! docker_ok; then
    msg="Docker is required and must be usable by this user (Linux: curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker \$USER, then log in again)"
    [ -n "${BENCHCAD_SKIP_SETUP:-}" ] && step "WARNING: $msg" || die "$msg"
fi
# BENCHCAD_SKIP_SETUP (tests only): skip uv sync, the image and the self-test.
SETUP=1; [ -n "${BENCHCAD_SKIP_SETUP:-}" ] && SETUP=0

# 2. the environment, and a self-test of what the scorers import (OCP, vtk, cadquery)
if [ "$SETUP" = 1 ]; then step "uv sync"; uv sync --frozen --quiet --group harness; fi   # the provider SDKs live in the harness group
PY=(uv run --no-sync python)
[ -z "${BENCHCAD_PYTHON:-}" ] || PY=("$BENCHCAD_PYTHON")
SELFTEST='import OCP.TopoDS, vtk, cadquery, trimesh, envs.verifiers.part, envs.verifiers.assembly, envs.verifiers.ecad'
selftest() { "${PY[@]}" -c "$SELFTEST" > "${TMPDIR:-/tmp}/run_core_selftest.log" 2>&1; }
if [ "$SETUP" = 1 ] && ! selftest; then
    # Measured in a clean ubuntu:24.04: OCP (the CAD kernel) needs libGL.so.1, which is the only
    # system library missing there; with libgl1 every scorer imports and every example scores 1.0.
    if grep -q "libGL.so" "${TMPDIR:-/tmp}/run_core_selftest.log" && command -v apt-get >/dev/null 2>&1; then
        SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo -n"
        if [ -z "$SUDO" ] || sudo -n true 2>/dev/null; then
            step "installing libgl1 (OCP needs libGL.so.1)"
            $SUDO apt-get -o DPkg::Lock::Timeout=180 update -qq >/dev/null 2>&1 || true
            $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=180 install -y -qq libgl1 >/dev/null 2>&1 || true
        fi
    fi
    selftest || { tail -3 "${TMPDIR:-/tmp}/run_core_selftest.log" >&2
                  die "the scorers cannot load (above). On Debian/Ubuntu: sudo apt-get install -y libgl1 -- then run again"; }
fi
[ "$SETUP" = 1 ] && step "scorer self-test passed (OCP, vtk, cadquery)"

# 3. the sandbox image; the model's code never runs outside it
case "$(uname -m)" in arm64|aarch64) ARCH=arm64 ;; *) ARCH=amd64 ;; esac
export CADENV_DOCKER_IMAGE="${CADENV_DOCKER_IMAGE:-benchcad-sandbox:$ARCH}"
unset CADENV_LOCAL
if ! docker image inspect "$CADENV_DOCKER_IMAGE" >/dev/null 2>&1; then
    if [ "$SETUP" = 0 ]; then step "would build $CADENV_DOCKER_IMAGE from sandbox/"
    else step "building $CADENV_DOCKER_IMAGE (once)"; docker build -q -t "$CADENV_DOCKER_IMAGE" sandbox/ >/dev/null; fi
fi

# 4. the dataset: downloaded once, checked every time
verify() {
    [ -f "$DATA/MANIFEST.sha256" ] || return 1
    if command -v sha256sum >/dev/null 2>&1; then (cd "$DATA" && sha256sum -c --quiet MANIFEST.sha256)
    else (cd "$DATA" && shasum -a 256 -c --quiet MANIFEST.sha256); fi
}
hf_access() {   # 0 when this account can read the gated repo (one metadata request, no download)
    local tok="${HF_TOKEN:-}"
    [ -n "$tok" ] || tok=$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null || true)
    [ -n "$tok" ] || return 2
    [ "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $tok" \
        "https://huggingface.co/api/datasets/$DATASET_REPO/tree/main")" = 200 ]
}
HAVE_DATA=1
if ! verify >/dev/null 2>&1; then
    if [ "$DRY" = 1 ]; then
        HAVE_DATA=0
        rc=0; hf_access || rc=$?
        case $rc in
            0) step "dataset not downloaded yet; Hugging Face access to $DATASET_REPO: OK" ;;
            2) die "no Hugging Face token: set HF_TOKEN (an account with access to $DATASET_REPO)" ;;
            *) die "this Hugging Face account cannot read $DATASET_REPO: request access at https://benchcad.com/access.html" ;;
        esac
    else
    step "downloading $DATASET_REPO to $DATA"
    mkdir -p "$DATA"
    HF_HUB_DISABLE_PROGRESS_BARS=1 uvx --quiet --from huggingface_hub hf download "$DATASET_REPO" --repo-type dataset \
        --local-dir "$DATA" > "$DATA.download.log" 2>&1 \
        || { tail -5 "$DATA.download.log" >&2; die "download failed: $DATASET_REPO is gated -- request access on its Hugging Face page, then set HF_TOKEN (or run: uvx --from huggingface_hub hf auth login)"; }
    verify || die "$DATA does not match MANIFEST.sha256; delete it and run again"
    fi
fi
[ "$HAVE_DATA" = 1 ] && step "dataset verified: $DATA"

# 5. preflight, before any spend: dataset version, scorer seal, model, efforts, keys
EFFORTS=$("${PY[@]}" - "$MODEL" "$EFFORTS" "$DATA" "$DATASET_NAME" "$DATASET_VERSION" "${CASES:-$DATA}" "$HAVE_DATA" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "harness"); sys.path.insert(0, ".")
import run as R
from envs.common.ecad_graph.spatial_reference import scorer_digest
model, efforts, data, name, version, cases, have = sys.argv[1:]
ds = json.loads((Path(data) / "dataset.json").read_text()) if have == "1" else \
    {"name": name, "version": version, "scorer_digest": scorer_digest(), "n_cases": None}
if (ds.get("name"), ds.get("version")) != (name, version):
    sys.exit(f"run_core: this harness expects {name} {version}, the download is {ds.get('name')} {ds.get('version')}; "
             f"git pull, or delete {data} and run again")
if ds.get("scorer_digest") != scorer_digest():
    sys.exit("run_core: the dataset is bound to a different T6 scorer than this checkout; git pull")
n = sum(1 for _ in Path(data).rglob("case.json")) if have == "1" else None
if n != ds.get("n_cases"):
    sys.exit(f"run_core: {n} cases on disk, dataset.json says {ds.get('n_cases')}")
prefix, prov, _ = R.split_model(model)
sdk = {"anthropic": "anthropic", "openai_compat": "openai", "gemini": "google.genai"}.get(prov.kind)
if sdk:
    try:
        __import__(sdk)
    except ImportError:
        sys.exit(f"run_core: the {sdk} SDK is not installed; run: uv sync --frozen --group harness")
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

# Another run_core on this machine shares the memory: warn, and budget half of it.
LOCK="$(dirname "$DATA")/run_core.$(id -u).lock"
SHARED=0
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null && [ "$(cat "$LOCK")" != $$ ]; then
    SHARED=1; step "WARNING: another run_core is running here (pid $(cat "$LOCK")); budgeting half the memory"
fi
if [ "$DRY" = 0 ]; then mkdir -p "$(dirname "$LOCK")"; echo $$ > "$LOCK"; trap '[ "$(cat "$LOCK" 2>/dev/null)" = $$ ] && rm -f "$LOCK"' EXIT; fi
read -r AUTO_W SCORERS EXECS BUDGET <<< "$(mem_budget)"
if [ -z "$WORKERS" ]; then
    WORKERS=$AUTO_W
    step "workers $WORKERS per effort, auto: $BUDGET (--workers N overrides)"
else
    step "workers $WORKERS per effort (--workers override; the memory budget would size $AUTO_W), $EXECS sandbox executions and $SCORERS scorers at once"
fi
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
         --rounds "$ROUNDS" --rep 0 --workers "$WORKERS" --score-workers "$SCORERS" --max-execs "$EXECS"
         --resume --out "$OUT/$NAME.json"
         --work "work/core/$TAG/$NAME" ${EXTRA[@]+"${EXTRA[@]}"})
    FILES+=("$OUT/$NAME.json")
    if [ "$DRY" = 1 ]; then echo "[run_core] would run: ${CMD[*]}"; continue; fi
    step "effort $NAME: $OUT/$NAME.json (log $OUT/$NAME.log)"
    set +e
    "${CMD[@]}" 2>&1 | tee -a "$OUT/$NAME.log" | grep -E '^\s*\[[0-9]+/[0-9]+\]|cases ->'
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" = 137 ] || [ "$rc" = 9 ] || [ "$rc" = -9 ]; then
        die "run.py was killed (likely out of memory); re-run the same command to resume (it keeps finished cases); consider --workers N lower than $WORKERS"
    fi
    [ "$rc" = 0 ] || step "run.py exited $rc (see $OUT/$NAME.log); re-run the same command to resume"
done

# 7. the score
[ "$DRY" = 1 ] && { step "dry run: all checks passed"; exit 0; }
"${PY[@]}" tools/core_score.py "${FILES[@]}" --dataset "${CASES:-$DATA}" --dataset-info "$DATA" --json "$OUT/core_summary.json"
