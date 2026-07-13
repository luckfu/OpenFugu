# OpenFugu BFCL V4 运行文档

本文档说明如何使用 Berkeley Function Calling Leaderboard（BFCL V4）评测多个 worker，并训练 OpenFugu router。

## 1. 这套流程评测什么

每个 worker 收到完全相同的：

```text
用户问题 + JSON Schema function definitions
```

worker 必须通过原生 tool calling 返回函数名称和参数。OpenFugu 不自行判断答案，而是把结构化调用交给 BFCL V4 官方 `ast_checker`：

```text
worker 原生 tool_calls
  -> BFCL 官方 AST checker
  -> valid=true/false
  -> 每个 case × worker 的 0/1 分数
```

首批启用四个不需要 Docker 的类别：

```text
simple_python       400
parallel            200
multiple            200
parallel_multiple   200
总计               1000
```

BFCL 的参数值通常写在用户问题中，并允许多个合法答案或省略合法的可选参数。评测结果表示“模型输出是否符合 BFCL 官方标签”，不应直接解释成模型在所有真实场景中的绝对能力。少量自然语言用例可能存在歧义，因此不能根据十几条样本的单条输赢得出模型排名。

## 2. 环境要求

评测阶段：

```text
Python 3.10+
不需要 GPU
不需要 torch
不需要 Docker
需要 worker API key
```

训练阶段需要加载 `Qwen/Qwen3-0.6B`，建议至少 8GB 显存。

## 3. 准备配置

```bash
cp configs/bfcl.example.yaml configs/bfcl.yaml
```

编辑 `configs/bfcl.yaml`，重点检查 worker 顺序、模型名和 endpoint。不要把 key 写进 YAML：

```bash
export DEEPSEEK_API_KEY='...'
export ZHIPU_API_KEY='...'
```

`max_samples_per_category` 表示每个类别最多抽取多少条，`0` 表示使用该类别全部数据。调用数计算公式：

```text
用例数 = 类别数 × 每类采样数
调用数 = 用例数 × worker 数
```

建议分三个阶段运行。

冒烟测试，每类 3 条：

```text
4 类 × 3 条 = 12 cases
12 cases × 3 workers = 36 次调用
```

这一步只验证 API、原生 tool calling、官方评分器和断点文件能否正常工作，不适合比较模型能力或训练 router。

初步评测，每类 50 条：

```yaml
evaluation:
  max_samples_per_category: 50
```

```text
4 类 × 50 条 = 200 cases
200 cases × 3 workers = 600 次调用
```

这个规模可以减弱单条歧义样本的影响，适合检查模型差异和训练流程，但验证集仍然较小。

正式全量评测：

```yaml
evaluation:
  max_samples_per_category: 0
```

三个 worker 全量调用数为 `1000 × 3 = 3000`。

示例配置固定了 Gorilla/BFCL git revision。不要在同一轮多 worker 评测中改动它，否则不同 worker 的分数不再严格可比。

## 4. Dry-run

```bash
DRY_RUN=1 CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

它会：

```text
安装 litellm/pyyaml
浅克隆 Gorilla 仓库中的 BFCL 目录
加载官方 AST checker
枚举样本和调用数
```

它不会调用 worker API，也不会训练。

## 5. 正式评测

```bash
CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

正式执行前，每个 worker 都必须通过一个真实的原生 tool-call 预检。只返回普通文本、不支持 `tools` 参数或参数类型错误都会立即停止。

预检和正式评测错误会显示具体 worker、模型、endpoint、供应商错误码和资源提示。例如：

```text
[bfcl:preflight:error] worker=zhipu_glm_5_2
model=openai/glm-5.2
endpoint=https://open.bigmodel.cn/api/paas/v4/
类型=insufficient_quota
供应商错误码=1113
供应商消息=余额不足或无可用资源包,请充值。
需检查=智谱开放平台的账户余额或 glm-5.2 可用资源包
```

如果供应商只返回“无可用资源包”而没有具体套餐名称，程序会明确说明这一点，不能从 API 错误中推断出控制台里的具体资源包名称。

并发数在 YAML 中控制：

```yaml
evaluation:
  concurrency: 3
  request_timeout: 120
```

输出：

```text
openfugu_bfcl/bfcl_predictions.jsonl
openfugu_bfcl/bfcl_scores.csv
```

`predictions.jsonl` 每完成一次调用就立即追加，因此支持中断后继续。每一行是一个 `case × worker` 的完整记录，其中：

```text
valid=true
  BFCL 官方评分器判定通过

valid=false + provider_message 非空
  API、鉴权、额度、超时、schema 或通讯失败

valid=false + error_type 类似 parallel_function_checker_no_order:...
  API 调用成功，但 BFCL 官方评分器判定函数名、数量或参数不正确
```

错误日志会指出具体 worker 和供应商资源。供应商只返回“余额不足”时，程序只能提示账户余额或该模型资源包，无法知道控制台中未通过 API 返回的具体套餐名称。

## 6. 断点续跑和重试

继续未完成项：

```bash
CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

只重试执行失败项：

```bash
RETRY_FAILED=1 CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

它会重试鉴权、限流、超时、JSON 解析、非法 tool schema 和普通 API 调用失败。BFCL 官方评分器判错的能力样本不会重试，因为重复调用会改变同一模型的采样口径；即使 `temperature: 0`，远程服务也不保证每次完全一致。

全部重新评测并覆盖预测输出：

```bash
NO_RESUME=1 CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

### 改变采样配置后重新开始

修改以下任一内容后，不要沿用旧断点：

```text
categories
max_samples_per_category
seed
worker 名称或模型
BFCL revision
```

先删除上一轮评测与训练产物：

```bash
rm -f openfugu_bfcl/bfcl_predictions.jsonl \
      openfugu_bfcl/bfcl_scores.csv \
      openfugu_bfcl/bfcl_worker_matrix.csv \
      openfugu_bfcl/trinity_bfcl.npy \
      openfugu_bfcl/trinity_bfcl.json
```

然后先确认规模，再正式运行：

```bash
DRY_RUN=1 CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

只因进程中断而继续相同实验时，不要删除文件，直接执行原命令即可。

## 7. 如何阅读评测结果

快速查看每个 worker 的通过率：

```bash
python - <<'PY'
import csv
from collections import defaultdict

rows = list(csv.DictReader(open("openfugu_bfcl/bfcl_scores.csv", encoding="utf-8")))
grouped = defaultdict(list)
for row in rows:
    grouped[row["worker"]].append(float(row["score"]))
for worker, scores in grouped.items():
    print(f"{worker}: {sum(scores):.0f}/{len(scores)} = {sum(scores)/len(scores):.2%}")
PY
```

查看所有未通过记录：

```bash
python - <<'PY'
import json

for line in open("openfugu_bfcl/bfcl_predictions.jsonl", encoding="utf-8"):
    row = json.loads(line)
    if not row.get("valid"):
        print(row["case_id"], row["worker"], row.get("error_type"), row.get("error"))
PY
```

比较模型时至少同时查看：总体通过率、各类别通过率、执行失败数和官方评分失败数。服务不可用是否计入能力取决于实验目标；当前 OpenFugu 流程会把最终未恢复的调用失败记为 `0` 分。

## 8. GPU 训练 router

把评测结果提交到 GitHub 后，在 GPU 服务器执行：

```bash
git pull
cp configs/bfcl.example.yaml configs/bfcl.yaml
# 确保 workers 名称和顺序与评测时完全一致
CONFIG_FILE=configs/bfcl.yaml bash scripts/colab_bfcl_router.sh
```

训练器会：

```text
读取 case × worker 的 BFCL 0/1 矩阵
划分训练集和验证集
用 Qwen3-0.6B 提取每个 case 的 hidden state
用 CMA-ES 训练 bias-free linear router head
报告验证集 router / 最佳单模型 / oracle
```

输出：

```text
openfugu_bfcl/bfcl_worker_matrix.csv
openfugu_bfcl/trinity_bfcl.npy
openfugu_bfcl/trinity_bfcl.json
```

`trinity_bfcl.json` 固化 worker slot 顺序和验证集指标。部署时必须使用相同顺序。

训练前建议至少完成每类 50 条。只有 12 个用例时，单条结果会影响 `8.3` 个百分点，训练集和验证集都不足以证明 router 能泛化。

## 9. 云服务器一把跑完

```bash
CONFIG_FILE=configs/bfcl.yaml bash scripts/cloud_bfcl_full_pipeline.sh
```

已有评测结果，只训练：

```bash
SKIP_EVAL=1 CONFIG_FILE=configs/bfcl.yaml bash scripts/cloud_bfcl_full_pipeline.sh
```

只评测，不训练：

```bash
SKIP_TRAIN=1 CONFIG_FILE=configs/bfcl.yaml bash scripts/cloud_bfcl_full_pipeline.sh
```

## 10. 提交产物

```bash
git add openfugu_bfcl/bfcl_predictions.jsonl \
        openfugu_bfcl/bfcl_scores.csv \
        openfugu_bfcl/bfcl_worker_matrix.csv \
        openfugu_bfcl/trinity_bfcl.json
git add -f openfugu_bfcl/trinity_bfcl.npy
git commit -m "Add BFCL evaluation and router head"
git push origin main
```
