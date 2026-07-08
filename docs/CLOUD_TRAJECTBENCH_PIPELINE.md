# 云服务器端到端运行 TRAJECT-Bench

本文档用于在一台新的云服务器上，从模型评估一路跑到 OpenFugu router 训练。

目标流程：

```text
clone OpenFugu
  ↓
配置 worker API
  ↓
跑 TRAJECT-Bench 评估
  ↓
重试失败项
  ↓
重算官方兼容分数
  ↓
训练 Qwen hidden-state router
  ↓
提交评估产物和 router head
```

## 1. 服务器建议

评估阶段不需要 GPU，但训练阶段需要加载 `Qwen/Qwen3-0.6B`。

建议：

```text
Python: 3.10 - 3.12
GPU: T4 / L4 / A10 / A100 均可
显存: 8GB+ 基本够用
磁盘: 20GB+
```

如果只跑评估，不训练，可以没有 GPU。

## 2. 拉取仓库

```bash
git clone https://github.com/luckfu/OpenFugu.git
cd OpenFugu
```

如果你是继续已有目录：

```bash
git pull --ff-only
```

## 3. 准备配置

复制本地配置：

```bash
cp configs/trajectbench.example.yaml configs/trajectbench.yaml
```

编辑：

```bash
nano configs/trajectbench.yaml
```

重点检查：

```yaml
evaluation:
  max_samples_per_domain: 1   # 先小跑；确认后再调大
  concurrency: 2
  request_timeout: 120
  max_tokens: 8192

workers:
  - name: deepseek-v4-flash
    model: openai/deepseek-chat
    api_base: https://api.deepseek.com/v1
    api_key: env:DEEPSEEK_API_KEY
```

`workers[*].name` 是 router slot 名称，后续部署时必须保持同样顺序。

## 4. 设置 API key

不要把 key 写进 YAML，使用环境变量：

```bash
export DEEPSEEK_API_KEY='你的 DeepSeek key'
export ZHIPU_API_KEY='你的智谱 key'
```

检查：

```bash
python - <<'PY'
import os
for k in ["DEEPSEEK_API_KEY", "ZHIPU_API_KEY"]:
    v = os.environ.get(k)
    print(k, "set" if v else "missing", len(v) if v else 0)
PY
```

## 5. 一键端到端运行

```bash
CONFIG_FILE=configs/trajectbench.yaml bash scripts/cloud_trajectbench_full_pipeline.sh
```

脚本会执行：

```text
1. dry-run 检查配置和样本
2. 正式评估 worker
3. RETRY_FAILED 重试失败项
4. 根据 predictions 重新计算官方兼容分数
5. 训练 router
6. 打印提交命令
```

默认失败重试 1 轮。可以改：

```bash
RETRY_ROUNDS=2 CONFIG_FILE=configs/trajectbench.yaml bash scripts/cloud_trajectbench_full_pipeline.sh
```

## 6. 常用开关

只评估，不训练：

```bash
SKIP_TRAIN=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/cloud_trajectbench_full_pipeline.sh
```

已有评估结果，只重新计算分数并训练：

```bash
SKIP_EVAL=1 SKIP_RETRY=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/cloud_trajectbench_full_pipeline.sh
```

跳过失败重试：

```bash
SKIP_RETRY=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/cloud_trajectbench_full_pipeline.sh
```

只跑评估脚本 dry-run：

```bash
DRY_RUN=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

## 7. 输出产物

评估输出：

```text
openfugu_trajectbench/trajectbench_predictions.jsonl
openfugu_trajectbench/trajectbench_scores.csv
openfugu_trajectbench/trajectbench_step_samples.jsonl
```

训练输出：

```text
openfugu_trajectbench/trajectbench_step_matrix.csv
openfugu_trajectbench/trinity_trajectbench.npy
```

其中：

```text
trinity_trajectbench.npy 是 router head
trajectbench_step_matrix.csv 是训练用分数矩阵
```

## 8. 查看训练结果

训练日志里会出现：

```text
[基线] zero=... 最佳单模型=... 理论上限=...
[结果] 训练分数=... 最佳单模型=... 理论上限=...
```

含义：

```text
最佳单模型: 永远选择平均分最高的 worker
理论上限: 每个 step 都选择真实最高分 worker 的 oracle
训练分数: router 根据 Qwen hidden state 选择 worker 的结果
```

如果训练分数高于最佳单模型，说明 router 学到了有效路由信号。

## 9. 提交结果

先配置 git 身份：

```bash
git config user.name "luckfu"
git config user.email "你的 GitHub 邮箱"
```

提交：

```bash
git add openfugu_trajectbench/trajectbench_predictions.jsonl \
        openfugu_trajectbench/trajectbench_scores.csv \
        openfugu_trajectbench/trajectbench_step_samples.jsonl \
        openfugu_trajectbench/trajectbench_step_matrix.csv
git add -f openfugu_trajectbench/trinity_trajectbench.npy
git commit -m "Update TRAJECT-Bench evaluation and router head"
git push origin main
```

GitHub 不支持账号密码 push。HTTPS push 时，Password 要填 GitHub Personal Access Token。

## 10. 注意事项

`TRAJECT-Bench/` 是第三方 checkout，默认不提交。

`configs/trajectbench.yaml` 被 `.gitignore` 忽略，可以保存你的本地 worker 配置，但不要提交 key。

服务质量也计入结果：

```text
empty_response
rate limit
timeout
connection error
```

这些失败会按 0 分进入训练数据，因为对 router 来说，稳定性也是 worker 能力的一部分。
