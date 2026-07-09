#!/usr/bin/env bash
set -euo pipefail

# 云服务器端到端流程：TRAJECT-Bench 评估 -> 失败重试 -> 重算分数 -> 训练 router。
#
# 用法：
#   CONFIG_FILE=configs/trajectbench.yaml bash scripts/cloud_trajectbench_full_pipeline.sh
#
# 常用开关：
#   SKIP_EVAL=1       跳过评估，只重算和训练
#   SKIP_RETRY=1      跳过失败重试
#   SKIP_TRAIN=1      跳过训练
#   RETRY_ROUNDS=2    失败重试轮数

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
CONFIG_FILE="${CONFIG_FILE:-configs/trajectbench.yaml}"
RETRY_ROUNDS="${RETRY_ROUNDS:-1}"

log() {
  printf '\n\033[1;35m[cloud-traject]\033[0m %s\n' "$*"
}

die() {
  printf '\n\033[1;31m[cloud-traject:error]\033[0m %s\n' "$*" >&2
  exit 1
}

cd "$OPENFUGU_DIR"
log "OpenFugu 目录: $OPENFUGU_DIR"
log "配置文件: $CONFIG_FILE"

[[ -f "$CONFIG_FILE" ]] || die "找不到配置文件。先执行: cp configs/trajectbench.example.yaml configs/trajectbench.yaml"

log "环境检查"
python --version
git --version
python -m pip install -q pyyaml
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  log "未检测到 nvidia-smi；评估可以继续，但训练 Qwen router 会很慢。"
fi

log "检查 worker API key 环境变量"
python - "$CONFIG_FILE" <<'PY'
import json, os, sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if path.suffix.lower() == ".json":
    cfg = json.loads(text)
else:
    import yaml
    cfg = yaml.safe_load(text) or {}

missing = []
for worker in cfg.get("workers", []):
    ref = worker.get("api_key")
    if isinstance(ref, str) and ref.startswith("env:"):
        name = ref.split(":", 1)[1]
        if not os.environ.get(name):
            missing.append((worker.get("name") or worker.get("model"), name))

if missing:
    print("缺少环境变量:")
    for worker, name in missing:
        print(f"  {worker}: {name}")
    raise SystemExit(2)

print("API key 环境变量检查通过")
PY

if [[ "${SKIP_EVAL:-0}" != "1" && "${SKIP_EVAL:-false}" != "true" ]]; then
  log "阶段 1/5: dry-run 检查配置和样本"
  DRY_RUN=1 CONFIG_FILE="$CONFIG_FILE" bash scripts/prepare_trajectbench.sh

  log "阶段 2/5: 正式评估 worker"
  CONFIG_FILE="$CONFIG_FILE" bash scripts/prepare_trajectbench.sh
else
  log "跳过评估阶段"
fi

if [[ "${SKIP_RETRY:-0}" != "1" && "${SKIP_RETRY:-false}" != "true" ]]; then
  for i in $(seq 1 "$RETRY_ROUNDS"); do
    log "阶段 3/5: 重试失败项，第 $i/$RETRY_ROUNDS 轮"
    RETRY_FAILED=1 CONFIG_FILE="$CONFIG_FILE" bash scripts/prepare_trajectbench.sh
  done
else
  log "跳过失败重试阶段"
fi

log "阶段 4/5: 使用当前 predictions 调用官方 parser/metrics 重新计算分数"
TRAJECTBENCH_DIR="$(python - "$CONFIG_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
if path.suffix.lower() == ".json":
    cfg = json.loads(text)
else:
    import yaml
    cfg = yaml.safe_load(text) or {}
tb = cfg.get("trajectbench") or {}
print(tb.get("dir") or "TRAJECT-Bench")
PY
)"
python eval/eval_trajectbench.py --config "$CONFIG_FILE" --trajectbench-dir "$TRAJECTBENCH_DIR" --recompute-only

if [[ "${SKIP_TRAIN:-0}" != "1" && "${SKIP_TRAIN:-false}" != "true" ]]; then
  log "阶段 5/5: 训练 TRAJECT-Bench router"
  CONFIG_FILE="$CONFIG_FILE" bash scripts/colab_trajectbench_router.sh
else
  log "跳过训练阶段"
fi

log "最终产物"
ls -lh openfugu_trajectbench || true

log "建议提交命令"
cat <<'EOF'
git status
git add openfugu_trajectbench/trajectbench_predictions.jsonl \
        openfugu_trajectbench/trajectbench_scores.csv \
        openfugu_trajectbench/trajectbench_step_samples.jsonl \
        openfugu_trajectbench/trajectbench_step_matrix.csv
git add -f openfugu_trajectbench/trinity_trajectbench.npy
git commit -m "Update TRAJECT-Bench evaluation and router head"
git push origin main
EOF

log "完成"
