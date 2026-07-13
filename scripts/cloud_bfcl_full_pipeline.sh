#!/usr/bin/env bash
set -euo pipefail

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
CONFIG_FILE="${CONFIG_FILE:-configs/bfcl.yaml}"
RETRY_ROUNDS="${RETRY_ROUNDS:-1}"

log() { printf '\n\033[1;35m[cloud-bfcl]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[cloud-bfcl:error]\033[0m %s\n' "$*" >&2; exit 1; }

cd "$OPENFUGU_DIR"
[[ -f "$CONFIG_FILE" ]] || die "找不到 $CONFIG_FILE；先复制 configs/bfcl.example.yaml"

log "检查 worker API key 环境变量"
python -m pip install -q pyyaml
python - "$CONFIG_FILE" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(sys.argv[1]); text = p.read_text(encoding="utf-8")
if p.suffix.lower() == ".json":
    cfg = json.loads(text)
else:
    import yaml
    cfg = yaml.safe_load(text) or {}
missing = []
for worker in cfg.get("workers") or []:
    value = worker.get("api_key")
    if isinstance(value, str) and value.startswith("env:"):
        name = value.split(":", 1)[1]
        if not os.environ.get(name):
            missing.append(f"{worker.get('name') or worker.get('model')}: {name}")
if missing:
    raise SystemExit("缺少环境变量:\n  " + "\n  ".join(missing))
print("API key 检查通过")
PY

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  log "阶段 1/3: dry-run"
  DRY_RUN=1 CONFIG_FILE="$CONFIG_FILE" bash scripts/prepare_bfcl.sh
  log "阶段 2/3: 正式评测"
  CONFIG_FILE="$CONFIG_FILE" bash scripts/prepare_bfcl.sh
  for round in $(seq 1 "$RETRY_ROUNDS"); do
    log "重试失败项 $round/$RETRY_ROUNDS"
    RETRY_FAILED=1 CONFIG_FILE="$CONFIG_FILE" bash scripts/prepare_bfcl.sh
  done
else
  log "跳过评测，复用已有 predictions"
fi

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  log "阶段 3/3: 训练 router"
  CONFIG_FILE="$CONFIG_FILE" bash scripts/colab_bfcl_router.sh
else
  log "跳过训练"
fi

log "输出文件"
ls -lh openfugu_bfcl || true
