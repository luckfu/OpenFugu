#!/usr/bin/env bash
set -euo pipefail

# 一次性搭建统一评估器环境（EvalScope + BFCL 官方评分器）。
# 独立虚拟环境，不污染主环境；所有版本钉死，换机器直接重跑本脚本即可。
#
# 用法:
#   bash scripts/setup_eval_env.sh
#   EVAL_ENV_DIR=.venv-eval bash scripts/setup_eval_env.sh
#
# 之后评测统一走:
#   .venv-eval/bin/evalscope eval --model <模型> --api-url <端点> --api-key ... \
#     --eval-type openai_api --datasets <基准> --limit <条数> --work-dir <输出目录>

OPENFUGU_DIR="${OPENFUGU_DIR:-$(pwd)}"
EVAL_ENV_DIR="${EVAL_ENV_DIR:-.venv-eval}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

log() { printf '\n\033[1;34m[eval-env]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[eval-env:error]\033[0m %s\n' "$*" >&2; exit 1; }

cd "$OPENFUGU_DIR"

if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip install --python "$EVAL_ENV_DIR/bin/python")
  if [[ ! -x "$EVAL_ENV_DIR/bin/python" ]]; then
    log "创建虚拟环境 $EVAL_ENV_DIR (uv, Python $PYTHON_VERSION)"
    uv venv "$EVAL_ENV_DIR" --python "$PYTHON_VERSION"
  fi
else
  PIP=("$EVAL_ENV_DIR/bin/python" -m pip install -q)
  if [[ ! -x "$EVAL_ENV_DIR/bin/python" ]]; then
    log "创建虚拟环境 $EVAL_ENV_DIR (python -m venv)"
    python3 -m venv "$EVAL_ENV_DIR"
    "$EVAL_ENV_DIR/bin/python" -m pip install -q --upgrade pip
  fi
fi

# 注意安装顺序与钉版原因（在 macOS 12 / Python 3.12 上实测可用的组合）：
# 1. 不要用 `evalscope[bfcl]` extra：pip 解析会把 evalscope 回退到 1.5.x。
# 2. bfcl-eval 必须 --no-deps：它钉 tree_sitter==0.21.3 且依赖 faiss-cpu，
#    faiss-cpu 没有 macOS 14 以下的轮子（只有 memory_* 类别需要 faiss）。
# 3. bfcl_eval 的 model_config 会 import 全部 provider handler，
#    因此即使只用 OpenAI 兼容 worker 也要装齐 provider SDK。
# 4. mistralai 钉 1.x：2.x 移除了顶层 Mistral 类，bfcl-eval 尚未适配。
log "安装 EvalScope 主体"
"${PIP[@]}" 'evalscope==1.9.1'

# ifeval 等基准需要额外依赖（evalscope[ifeval] 才带），提前装好避免下载/评测时报 ImportError
log "安装基准附加依赖（ifeval 等）"
"${PIP[@]}" langdetect immutabledict nltk

log "安装 BFCL 官方评分器（--no-deps 跳过 faiss-cpu）"
"${PIP[@]}" --no-deps 'bfcl-eval==2025.10.27.1'

log "安装 bfcl-eval 运行必需依赖（钉版）"
"${PIP[@]}" \
  'tree_sitter==0.21.3' \
  'tree-sitter-java==0.21.0' \
  'tree-sitter-javascript==0.21.4' \
  'tenacity==9.1.4' \
  'anthropic==0.120.2' \
  'cohere==7.0.8' \
  'google-genai==2.14.0' \
  'mistralai==1.9.11' \
  'boto3==1.43.58' \
  'writerai==4.0.1' \
  'qwen-agent==0.0.34' \
  'soundfile==0.14.0' \
  'datamodel-code-generator==0.71.0'

log "自检：evalscope 与 BFCL 官方评分链路"
"$EVAL_ENV_DIR/bin/python" - <<'PY'
import evalscope
assert evalscope.__version__ == "1.9.1", evalscope.__version__
from bfcl_eval.eval_checker.eval_runner import *  # noqa: F401,F403 触发全量 import 检查
print(f"evalscope={evalscope.__version__} bfcl_eval 评分链路 import OK")
PY

log "完成。评测入口: $EVAL_ENV_DIR/bin/evalscope"
