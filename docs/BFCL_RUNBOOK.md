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

BFCL 的参数值写在用户问题中，并允许多个合法答案或省略合法的可选参数，避免用隐藏参数惩罚模型。

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

示例配置默认每类只抽 3 条，因此是：

```text
4 类 × 3 条 = 12 cases
12 cases × 3 workers = 36 次调用
```

全量评测改为：

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

`predictions.jsonl` 每完成一次调用就立即追加，因此支持中断后继续。

## 6. 断点续跑和重试

继续未完成项：

```bash
CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

重试未通过项：

```bash
RETRY_FAILED=1 CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

全部重新评测并覆盖输出：

```bash
NO_RESUME=1 CONFIG_FILE=configs/bfcl.yaml bash scripts/prepare_bfcl.sh
```

## 7. GPU 训练 router

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

## 8. 云服务器一把跑完

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

## 9. 提交产物

```bash
git add openfugu_bfcl/bfcl_predictions.jsonl \
        openfugu_bfcl/bfcl_scores.csv \
        openfugu_bfcl/bfcl_worker_matrix.csv \
        openfugu_bfcl/trinity_bfcl.json
git add -f openfugu_bfcl/trinity_bfcl.npy
git commit -m "Add BFCL evaluation and router head"
git push origin main
```
