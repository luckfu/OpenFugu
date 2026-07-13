# OpenFugu BFCL Multi-Turn 运行文档

本文档说明如何使用 BFCL 官方多轮环境评测多个 worker。该流程与单轮 BFCL 分开保存结果。

## 1. 它测试什么

单轮 BFCL 只执行一次模型请求。Multi-Turn 会在每个用户轮次内循环：

```text
当前对话和可用工具
  -> worker 返回 tool_calls
  -> BFCL 本地执行工具
  -> 执行结果写回对话
  -> 再次请求同一个 worker
  -> worker 不再调用工具，本轮结束
  -> 进入下一用户轮次
```

一个 episode 完成后，程序把全部函数调用交给 BFCL 官方 `multi_turn_checker`。官方评分器逐轮检查：

```text
工具是否成功执行
工具返回结果是否覆盖标准结果
有状态 API 的最终状态是否与标准状态一致
缺少参数时是否停止调用并等待用户补充
缺少函数时是否在函数被补充后继续执行
```

工具环境是 BFCL 仓库中的本地 Python API，不需要 Docker，也不会操作真实的航班、消息、工单或本机文件系统。

## 2. 当前支持的类别

```text
multi_turn_base           200 episodes
multi_turn_miss_func      200 episodes
multi_turn_miss_param     200 episodes
multi_turn_long_context   200 episodes
```

第一轮只配置 `multi_turn_base`。跑通后再逐类加入，便于定位接口和模型行为问题。

## 3. 准备配置

本机已经生成 `configs/bfcl_multiturn.yaml`。其他机器执行：

```bash
cp configs/bfcl_multiturn.example.yaml configs/bfcl_multiturn.yaml
```

设置 API key：

```bash
export DEEPSEEK_API_KEY='...'
export ZHIPU_API_KEY='...'
```

默认配置抽取 2 个 episode：

```yaml
evaluation:
  categories:
    - multi_turn_base
  max_samples_per_category: 2
  concurrency: 3
  max_steps_per_turn: 20
```

多轮评测不能提前精确计算 API 调用数。每个用户轮次至少调用一次模型；每执行一批工具后还会再次调用模型，直到模型结束当前轮次。

## 4. 检查数据，不调用模型

```bash
DRY_RUN=1 CONFIG_FILE=configs/bfcl_multiturn.yaml \
  bash scripts/prepare_bfcl_multiturn.sh
```

默认应显示：

```text
episode=2
worker=3
episode-worker 总数=6
```

`DRY_RUN=1` 会加载官方数据、函数文档和 checker，但不会预检或调用 worker。

## 5. 正式冒烟测试

```bash
CONFIG_FILE=configs/bfcl_multiturn.yaml \
  bash scripts/prepare_bfcl_multiturn.sh
```

进度按 `episode × worker` 统计，同时显示已经产生的真实模型调用数：

```text
[bfcl-mt] 进度 3/6 (50.0%) 通过=2 未通过=1 模型调用=24
```

输出文件：

```text
openfugu_bfcl_multiturn/bfcl_multiturn_predictions.jsonl
openfugu_bfcl_multiturn/bfcl_multiturn_scores.csv
openfugu_bfcl_multiturn/bfcl_multiturn_step_samples.jsonl
```

`predictions.jsonl` 保存完整 episode，`scores.csv` 用于查看官方通过率，`step_samples.jsonl` 将每次模型请求展开为一行。

## 6. 断点续跑

相同配置下进程中断，直接重新执行原命令：

```bash
CONFIG_FILE=configs/bfcl_multiturn.yaml \
  bash scripts/prepare_bfcl_multiturn.sh
```

已经完成的 `case × worker` 会跳过。正在执行但尚未写完的 episode 会从该 episode 开头重新执行，因为半截环境状态不能可靠恢复。

只重试鉴权、额度、限流、超时、解析和普通 API 调用失败：

```bash
RETRY_FAILED=1 CONFIG_FILE=configs/bfcl_multiturn.yaml \
  bash scripts/prepare_bfcl_multiturn.sh
```

BFCL 官方 checker 判错的能力失败不会重试。全部覆盖重跑：

```bash
NO_RESUME=1 CONFIG_FILE=configs/bfcl_multiturn.yaml \
  bash scripts/prepare_bfcl_multiturn.sh
```

修改类别、采样数、随机种子或 worker 后，建议清理旧结果再运行：

```bash
rm -rf openfugu_bfcl_multiturn
```

## 7. 扩大规模

冒烟测试通过后，先把 `multi_turn_base` 提高到 10 个 episode：

```yaml
evaluation:
  max_samples_per_category: 10
```

确认成本、耗时和供应商稳定性后，再提高到 50。全量 200 个 episode：

```yaml
evaluation:
  max_samples_per_category: 0
```

不要直接同时启用四类全量。多轮 episode 的模型调用次数远高于单轮，而且长上下文类别可能显著增加输入 token 成本。

## 8. 逐步路由数据说明

`bfcl_multiturn_step_samples.jsonl` 每行包含：

```text
case_id / worker / turn / step
context                 当前模型实际看到的完整对话
available_functions     当前可用工具
prediction              模型返回的 tool calls
execution_results       BFCL 本地工具结果
episode_valid           整个 episode 的官方结果
```

这些记录已经具备逐步路由所需的上下文，但 `episode_valid` 是整段奖励，不是精确的单步标签。不同 worker 可能在前面采取不同动作，导致后续环境状态不同，因此不能直接把相同 `turn/step` 的三行拼成 worker 得分矩阵。

后续训练前还需要增加 teacher-forced 采集：从同一官方状态出发，让所有 worker 对同一个下一步上下文作答，再用该步的官方预期调用评分。当前多轮评测先用于验证 worker 的完整 Agent 能力并保存原始轨迹。
