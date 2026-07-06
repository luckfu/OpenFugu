#!/usr/bin/env bash
set -euo pipefail

# Colab-oriented entrypoint for training an OpenFugu TRINITY router head on
# LiveClawBench rewards.
#
# Typical Colab usage:
#   !git clone https://github.com/luckfu/OpenFugu.git
#   %cd OpenFugu
#   !CONFIG_FILE=configs/liveclawbench_colab.example.yaml \
#     bash scripts/colab_liveclawbench_router.sh
#
# If multiple custom/... workers need different endpoints:
#   WORKER_AE='0:CUSTOM_BASE_URL=https://a.example/v1;0:CUSTOM_API_KEY=key-a;1:CUSTOM_BASE_URL=https://b.example/v1;1:CUSTOM_API_KEY=key-b'
#
# Important: LiveClawBench's official Harbor runner needs Docker. Some Colab
# runtimes do not expose a working Docker daemon; this script checks that early.

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
CONFIG_FILE="${CONFIG_FILE:-}"
LIVECLAWBENCH_DIR="${LIVECLAWBENCH_DIR:-}"
LIVECLAWBENCH_REPO="${LIVECLAWBENCH_REPO:-https://github.com/Mosi-AI/LiveClawBench.git}"

SLOT_MODELS="${SLOT_MODELS:-custom/model-a,custom/model-b}"
N_TRAIN="${N_TRAIN:-8}"
ITERS="${ITERS:-12}"
SIGMA0="${SIGMA0:-0.3}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
TIMEOUT_MULTIPLIER="${TIMEOUT_MULTIPLIER:-1.0}"
JOBS_DIR="${JOBS_DIR:-/content/openfugu_liveclawbench_jobs}"
OUT_HEAD="${OUT_HEAD:-/content/trinity_liveclawbench.npy}"
MATRIX_OUT="${MATRIX_OUT:-/content/liveclawbench_scores.csv}"

# Optional task filters. Examples:
#   DOMAINS="Coding & Software Dev|DevOps & Env Repair"
#   DIFFICULTIES="easy|medium"
#   INCLUDE_REGEX="vue|git|blog"
DOMAINS="${DOMAINS:-}"
DIFFICULTIES="${DIFFICULTIES:-}"
INCLUDE_REGEX="${INCLUDE_REGEX:-}"
WORKER_AE="${WORKER_AE:-}"
WORKER_EE="${WORKER_EE:-}"
WORKER_AK="${WORKER_AK:-}"

# Set to 0 if you want lazy scoring during CMA-ES instead of full task x worker
# precomputation. Precompute is slower up front but makes the training loop cheap
# and deterministic with respect to Harbor rewards.
PRECOMPUTE_ALL="${PRECOMPUTE_ALL:-1}"

log() {
  printf '\n\033[1;34m[colab-liveclaw]\033[0m %s\n' "$*"
}

die() {
  printf '\n\033[1;31m[colab-liveclaw:error]\033[0m %s\n' "$*" >&2
  exit 1
}

cd "$OPENFUGU_DIR"

if [[ -n "$CONFIG_FILE" && -z "$LIVECLAWBENCH_DIR" ]]; then
  LIVECLAWBENCH_DIR="$(python - "$CONFIG_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit
text = path.read_text()
if path.suffix.lower() == ".json":
    cfg = json.loads(text)
else:
    try:
        import yaml
        cfg = yaml.safe_load(text) or {}
    except Exception:
        cfg = {}
        for line in text.splitlines():
            if line.strip().startswith("liveclawbench_dir:"):
                cfg["liveclawbench_dir"] = line.split(":", 1)[1].strip().strip('"').strip("'")
                break
print(cfg.get("liveclawbench_dir") or "")
PY
)"
fi
LIVECLAWBENCH_DIR="${LIVECLAWBENCH_DIR:-/content/LiveClawBench}"

log "OpenFugu dir: $OPENFUGU_DIR"
if [[ -n "$CONFIG_FILE" ]]; then
  log "Config file: $CONFIG_FILE"
fi
log "LiveClawBench dir: $LIVECLAWBENCH_DIR"
if [[ -z "$CONFIG_FILE" ]]; then
  log "Slot models: $SLOT_MODELS"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  log "nvidia-smi not found; continuing, but Qwen3-0.6B will be slow without GPU."
fi

log "Installing OpenFugu Python dependencies"
python -m pip install -U pip
python -m pip install -r requirements.txt

log "Fetching OpenFugu artifacts"
python scripts/fetch_artifacts.py
export FUGU_VECTOR="$OPENFUGU_DIR/artifacts/model_iter_60.npy"

if [[ -z "${FUGU_MODEL:-}" ]]; then
  log "Resolving Qwen/Qwen3-0.6B snapshot path"
  FUGU_MODEL="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-0.6B"))
PY
)"
  export FUGU_MODEL
fi
log "FUGU_MODEL=$FUGU_MODEL"
log "FUGU_VECTOR=$FUGU_VECTOR"

if [[ ! -d "$LIVECLAWBENCH_DIR/.git" ]]; then
  log "Cloning LiveClawBench"
  rm -rf "$LIVECLAWBENCH_DIR"
  git clone --depth 1 "$LIVECLAWBENCH_REPO" "$LIVECLAWBENCH_DIR"
else
  log "LiveClawBench already exists; pulling latest shallow changes if possible"
  git -C "$LIVECLAWBENCH_DIR" pull --ff-only || true
fi

if ! command -v docker >/dev/null 2>&1; then
  die "Docker CLI is not available. LiveClawBench/Harbor requires Docker; use a Colab runtime with Docker support or run this on a VM."
fi

if ! docker info >/dev/null 2>&1; then
  die "Docker daemon is not running or not accessible. LiveClawBench/Harbor cannot run real verifier tasks without it."
fi

log "Setting up LiveClawBench / Harbor"
(
  cd "$LIVECLAWBENCH_DIR"
  ./setup.sh
)

HARBOR_BIN="$LIVECLAWBENCH_DIR/.venv/bin/harbor"
if [[ ! -x "$HARBOR_BIN" ]]; then
  HARBOR_BIN="$(command -v harbor || true)"
fi
[[ -n "$HARBOR_BIN" ]] || die "Could not find harbor after LiveClawBench setup"
log "Harbor: $HARBOR_BIN"

AE_ARGS=()
add_ae_var() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    AE_ARGS+=(--ae "${name}=${!name}")
  fi
}

# Global env for Harbor/OpenClaw. This is useful when all workers share one
# gateway, or when using native provider prefixes such as openai/... and
# anthropic/... .
for name in \
  CUSTOM_BASE_URL CUSTOM_API_KEY CUSTOM_CONTEXT_WINDOW CUSTOM_MAX_TOKENS CUSTOM_REASONING CUSTOM_API \
  OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY \
  OPENROUTER_API_KEY MOONSHOT_API_KEY VOLCANO_ENGINE_API_KEY DEEPSEEK_API_KEY
do
  add_ae_var "$name"
done

WORKER_ARGS=()
add_worker_items() {
  local flag="$1"
  local raw="$2"
  [[ -n "$raw" ]] || return 0
  local old_ifs="$IFS"
  IFS=';'
  read -ra items <<< "$raw"
  IFS="$old_ifs"
  for item in "${items[@]}"; do
    [[ -n "$item" ]] || continue
    WORKER_ARGS+=("$flag" "$item")
  done
}
add_worker_items --worker-ae "$WORKER_AE"
add_worker_items --worker-ee "$WORKER_EE"
add_worker_items --worker-ak "$WORKER_AK"

FILTER_ARGS=()
if [[ -n "$DOMAINS" ]]; then
  FILTER_ARGS+=(--domains "$DOMAINS")
fi
if [[ -n "$DIFFICULTIES" ]]; then
  FILTER_ARGS+=(--difficulties "$DIFFICULTIES")
fi
if [[ -n "$INCLUDE_REGEX" ]]; then
  FILTER_ARGS+=(--include-regex "$INCLUDE_REGEX")
fi

PRECOMPUTE_ARGS=()
if [[ "$PRECOMPUTE_ALL" == "1" || "$PRECOMPUTE_ALL" == "true" || "$PRECOMPUTE_ALL" == "yes" ]]; then
  PRECOMPUTE_ARGS+=(--precompute-all)
fi

log "Starting LiveClawBench router training"
if [[ -n "$CONFIG_FILE" ]]; then
  python train/train_trinity_liveclawbench.py \
    --config "$CONFIG_FILE" \
    --liveclawbench-dir "$LIVECLAWBENCH_DIR" \
    --harbor-bin "$HARBOR_BIN"
else
  python train/train_trinity_liveclawbench.py \
    --liveclawbench-dir "$LIVECLAWBENCH_DIR" \
    --router-model "$FUGU_MODEL" \
    --slot-models "$SLOT_MODELS" \
    --n-train "$N_TRAIN" \
    --iters "$ITERS" \
    --sigma0 "$SIGMA0" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --jobs-dir "$JOBS_DIR" \
    --out "$OUT_HEAD" \
    --matrix-out "$MATRIX_OUT" \
    --harbor-bin "$HARBOR_BIN" \
    --timeout-multiplier "$TIMEOUT_MULTIPLIER" \
    "${AE_ARGS[@]}" \
    "${WORKER_ARGS[@]}" \
    "${FILTER_ARGS[@]}" \
    "${PRECOMPUTE_ARGS[@]}"
fi

log "Done"
if [[ -z "$CONFIG_FILE" ]]; then
  log "Trained head: $OUT_HEAD"
  log "Score matrix: $MATRIX_OUT"
  log "Harbor jobs/cache: $JOBS_DIR"
else
  log "Output paths are controlled by $CONFIG_FILE"
fi
