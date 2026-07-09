# OpenFugu TRAJECT-Bench 运行文档

本文档记录 OpenFugu 使用 TRAJECT-Bench 生成 step/tool-call 级评测数据的流程。

当前阶段目标：

```text
本机完成 TRAJECT-Bench worker 评测
    ↓
生成 scores / predictions / step_samples
    ↓
后续再把数据拿到 GPU 服务器训练 router
```

注意：本阶段不训练 Qwen router，也不需要 GPU 或 torch。

## 1. 当前阶段做什么

TRAJECT-Bench 提供的是工具调用轨迹数据。我们用它来测试不同 worker 模型在“规划工具调用轨迹”上的表现。

TRAJECT-Bench 官方 README 里的标准入口是：

```bash
python evaluation/tool_evaluation_model.py \
  -model [model name] \
  -tool_select [tool selection mode] \
  -method [problem solving method] \
  -k [tool pool size] \
  -emb_model [embedding model] \
  -traj_type [trajectory type] \
  -traj_file [trajectory file name] \
  -log_dir [log directory] \
  -chk_dir [checkpoint directory] \
  -base_data_dir [base data directory]
```

我们没有直接用这个入口批量跑 worker，原因是官方脚本里的 `-model` 是白名单枚举，例如 `claude_v37`、`deepseek-chat`、`gemini-2.5-pro` 等。它不直接支持 OpenFugu YAML 里的任意 OpenAI-compatible 服务商配置，例如：

```yaml
workers:
  - name: zhipu_glm_5_2
    model: openai/glm-5.2
    api_base: https://open.bigmodel.cn/api/coding/paas/v4
    api_key: env:ZHIPU_API_KEY
```

所以 OpenFugu 使用自己的适配器：

```bash
python eval/eval_trajectbench.py --config configs/trajectbench.yaml --trajectbench-dir TRAJECT-Bench
```

但这个适配器现在做了两件事来对齐官方评测语义：

```text
Prompt: 默认读取 TRAJECT-Bench/evaluation/evaluation_prompt.json
Metric: 使用 TRAJECT-Bench 官方基础指标兼容实现
Tool pool: 支持 domain / all / fixed，默认 domain
```

也就是说，它不是另起一套 benchmark，而是用 LiteLLM 把“官方 prompt + 官方兼容指标”接到你的多 worker 配置上。

当前没有接官方 `retrieval` 工具池模式。原因是 OpenFugu 这一阶段的目标是比较不同 worker 在同一任务、同一工具池下的表现，生成路由训练标签；retrieval 本身会引入另一个“检索器质量”变量，后续需要单独评估。

本阶段会让每个 worker 看同一个任务，然后输出它认为应该调用的工具列表：

```text
query + available tools
    ↓
worker model
    ↓
predicted tool_list
    ↓
和 TRAJECT-Bench gold tool_list 对比打分
```

最后生成三类文件：

```text
openfugu_trajectbench/trajectbench_predictions.jsonl
openfugu_trajectbench/trajectbench_scores.csv
openfugu_trajectbench/trajectbench_step_samples.jsonl
```

这些文件会作为后续 router 训练的数据来源。

## 2. 准备配置文件

从 example 复制本地配置：

```bash
cp configs/trajectbench.example.yaml configs/trajectbench.yaml
```

`configs/trajectbench.yaml` 已经被 `.gitignore` 忽略，可以放本机路径和私有 worker 配置。

不要把 API key 直接写进 YAML。推荐用环境变量：

官方 CLI 参数在 OpenFugu YAML 里的对应关系：

```yaml
evaluation:
  method: direct        # 对应 -method direct
  tool_select: domain   # 对应 -tool_select domain
  k: 20                 # 对应 -k 20，仅 fixed 模式使用
  trajectory_types:
    - parallel          # 对应 -traj_type parallel
  trajectory_files:
    - simple_ver        # 对应 -traj_file simple_ver
```

```yaml
workers:
  - name: deepseek-v4-flash
    model: openai/deepseek-chat
    api_base: https://api.deepseek.com/v1
    api_key: env:DEEPSEEK_API_KEY
```

然后在终端设置：

```bash
export DEEPSEEK_API_KEY=...
export ZHIPU_API_KEY=...
```

## 3. 本机路径

本机运行时建议使用相对路径：

```yaml
trajectbench:
  dir: TRAJECT-Bench

outputs:
  dir: openfugu_trajectbench
  predictions_jsonl: openfugu_trajectbench/trajectbench_predictions.jsonl
  scores_csv: openfugu_trajectbench/trajectbench_scores.csv
  step_samples_jsonl: openfugu_trajectbench/trajectbench_step_samples.jsonl
```

不要在本机使用 `/content/...`，那是 Colab 路径。

## 4. 先做 dry-run

dry-run 用来检查配置和数据集，不会调用模型 API。

```bash
DRY_RUN=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

它会做：

```text
读取 configs/trajectbench.yaml
下载或更新 TRAJECT-Bench
读取 public_data
枚举要评测的样本
检查 worker 数量
打印前 20 个样本
```

它不会做：

```text
不会调用 DeepSeek / 智谱
不会消耗 API 费用
不会生成真实评分文件
不会训练 router
```

看到类似输出就说明 dry-run 通过：

```text
[trajectbench] TRAJECT-Bench dir: TRAJECT-Bench
[trajectbench] samples=145 workers=3
[trajectbench] Done
```

这里的含义是：

```text
145 个评测样本
3 个 worker
正式评测会产生 145 × 3 = 435 次模型调用
```

## 5. 小规模 API 试跑

第一次不要直接跑完整 435 次调用。建议先把配置缩小：

```yaml
evaluation:
  trajectory_types:
    - parallel
  trajectory_files:
    - simple_ver
  domains:
    - Travel
  max_samples_per_domain: 3
```

如果有 3 个 worker，则正式调用量是：

```text
3 samples × 3 workers = 9 次模型调用
```

先 dry-run 确认样本数量：

```bash
DRY_RUN=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

再正式跑：

```bash
CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

正式评测前脚本会先做 worker preflight：

```text
检查 api_key 环境变量是否存在
用每个 worker 发一个极小测试请求
如果某个 worker 认证失败，先停止，不继续消耗整批样本
```

如果只是想跳过 preflight：

```bash
SKIP_PREFLIGHT=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

不建议首次运行时跳过。

## 6. 正式评测

确认 API key、模型名、小规模试跑都正常后，恢复更大的评测范围，然后执行：

```bash
CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

正式评测会：

```text
读取每个 TRAJECT-Bench 样本
把 query + available tools 发给每个 worker
要求 worker 返回 tool_list JSON
解析 worker 输出
和 gold tool_list 计算分数
写入输出文件
```

运行时会显示进度摘要：

```text
[trajectbench] progress 120/870 ( 13.8%) ok=110 fail=10 skipped=20 remaining=750
```

默认每完成 10 条 worker-sample 输出一次；失败会立即输出。可以在配置里调整：

```yaml
evaluation:
  progress_every: 10
```

评测默认支持并发请求。公网模型单次请求通常会受模型排队、输出长度、provider 网关影响；如果串行跑，180 次调用会很慢。

```yaml
evaluation:
  concurrency: 3
```

建议从 `3` 开始。如果 provider 限流，再降到 `1`；如果账号额度允许，可以提高到 `6`。`sleep_seconds` 只在串行模式下生效。

慢请求可以设置超时：

```yaml
evaluation:
  request_timeout: 120
```

超过 120 秒的调用会记为 `timeout` 或 `call_failed`，后续可以用 `RETRY_FAILED=1` 单独重试失败项。

## 7. 断点续跑

评测脚本默认支持断点续跑。

每完成一个：

```text
sample_id + worker
```

脚本就会立即追加一行到：

```text
openfugu_trajectbench/trajectbench_predictions.jsonl
```

如果中途断网、超时、Ctrl+C 或机器重启，重新执行同一条命令即可：

```bash
CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

脚本会读取已有 `predictions.jsonl`，跳过已经完成的样本，只继续未完成部分。

看到类似输出表示 resume 生效：

```text
[trajectbench] resume enabled: loaded 120 completed worker-sample rows
[trajectbench] skipped completed rows: 120
```

### 重跑失败项

默认情况下，失败项也会被视为“已经完成”，避免无限重试。

如果你想只重试之前失败的项：

```bash
RETRY_FAILED=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

### 从头重跑

如果想忽略已有结果，从头开始：

```bash
NO_RESUME=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

注意：`NO_RESUME=1` 会覆盖输出文件，之前的预测结果不再用于跳过。

## 8. 输出文件说明

### predictions JSONL

路径：

```text
openfugu_trajectbench/trajectbench_predictions.jsonl
```

每一行是一条 worker 预测，包含：

```text
sample_id
worker
model
query
gold_tool_list
pred_tool_list
raw_response
score
error
```

这个文件适合排查模型输出格式问题。

### scores CSV

路径：

```text
openfugu_trajectbench/trajectbench_scores.csv
```

这是最方便人工查看的汇总表，包含：

```text
sample_id
worker
domain
trajectory_type
score
name_exact
inclusion
param_accuracy
error
```

其中：

```text
name_exact       工具集合是否完全匹配
inclusion        gold 工具有多少被 worker 命中
param_accuracy   工具参数是否匹配
score            综合分数
```

### step samples JSONL

路径：

```text
openfugu_trajectbench/trajectbench_step_samples.jsonl
```

这是后续 router 训练更关心的文件。它把一条工具轨迹拆成多个 step：

```text
sample_id
worker
domain
trajectory_type
step_index
query
prior_gold_tools
gold_tool
worker_selected_this_tool
trajectory_score
```

后续会基于这个文件继续构造：

```text
当前 step context -> 哪个 worker 更适合
```

## 9. 当前阶段不做什么

当前 TRAJECT-Bench 评测阶段不做：

```text
不加载 Qwen3-0.6B
不取 hidden state
不训练 model_iter_60.npy
不需要 GPU
不需要 torch
```

这些会在下一阶段，也就是“用评测结果训练 OpenFugu router”时处理。

## 10. 常见问题

### 为什么 dry-run 没有输出 scores？

因为 dry-run 只检查配置，不调用模型，不产生真实评分。

### 为什么本机不能用 /content/TRAJECT-Bench？

`/content` 是 Colab 路径。本机应使用：

```text
TRAJECT-Bench
```

### 为什么脚本默认不安装 requirements.txt？

因为本阶段只需要：

```text
litellm
pyyaml
```

完整 `requirements.txt` 包含 torch、transformers 等训练依赖，应该留到 GPU 服务器训练阶段安装。

如需强制安装完整依赖：

```bash
INSTALL_FULL_REQUIREMENTS=1 CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

### 智谱报“身份验证失败”怎么办？

这说明请求已经发到智谱 OpenAI-compatible endpoint，但 API key 没通过。

先确认当前 shell 里真的导出了环境变量：

```bash
python - <<'PY'
import os
for k in ["DEEPSEEK_API_KEY", "ZHIPU_API_KEY"]:
    v = os.environ.get(k)
    print(k, "set" if v else "missing", len(v) if v else 0)
PY
```

然后重新运行：

```bash
CONFIG_FILE=configs/trajectbench.yaml bash scripts/prepare_trajectbench.sh
```

正式评测会先做 preflight，小请求也失败时不会继续刷整批样本。

## 11. 在 GPU / Colab 上训练 router

本地评测完成并提交 `openfugu_trajectbench/` 后，可以在 Colab 或 GPU 服务器继续训练。

Colab 端流程：

```bash
git clone https://github.com/luckfu/OpenFugu.git
cd OpenFugu
```

确认评测产物已经随仓库存在：

```bash
ls openfugu_trajectbench/
```

应能看到：

```text
trajectbench_scores.csv
trajectbench_step_samples.jsonl
trajectbench_predictions.jsonl
```

然后训练 TRAJECT-Bench step-level router：

```bash
CONFIG_FILE=configs/trajectbench.example.yaml bash scripts/colab_trajectbench_router.sh
```

训练脚本会：

```text
安装训练依赖
下载 Qwen/Qwen3-0.6B
读取 openfugu_trajectbench/trajectbench_step_samples.jsonl
用 Qwen hidden state 提取每个 step context 的向量
用 sep-CMA-ES 训练 worker router head
```

输出：

```text
openfugu_trajectbench/trinity_trajectbench.npy
openfugu_trajectbench/trajectbench_step_matrix.csv
```

如果想改训练轮数：

```yaml
training:
  iters: 40
  sigma0: 0.3
  seed: 42
```

worker 顺序来自配置文件的 `workers` 列表。这个顺序就是 router 输出 slot 的顺序，必须和后续部署 worker pool 的顺序一致。

## 12. 下一步

本阶段完成后，下一步是：

```text
检查 scores.csv
确认 worker 输出质量和错误率
把 step_samples.jsonl 设计成 router 训练输入
在 GPU 服务器加载 Qwen3-0.6B 提取 hidden state
训练 step-level router head
```
