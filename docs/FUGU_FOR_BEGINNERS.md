# OpenFugu 科普导览：Qwen3-0.6B 被用来做了什么

这份文档面向刚接触 LLM 推理、hidden state、prefill、路由器这些概念的读者。它不追求论文级严谨，而是把 OpenFugu 这个项目从“用户发出一个问题”开始讲清楚。

核心结论先放前面：

> OpenFugu 不是把 Qwen3-0.6B 训练成一个更会聊天的大模型，而是把它用作一个很小的“调度器大脑”：读懂当前问题，产生一个内部向量，然后决定该让哪个外部 worker 模型来回答。

---

## 1. 项目在解决什么问题

普通调用大模型时，流程大概是：

```text
用户问题
  ↓
某一个模型，例如 GPT / Claude / Gemini / Qwen
  ↓
模型直接生成回答
```

OpenFugu 的思路不同。它假设不同模型各有擅长领域：

- 有的模型更擅长代码。
- 有的模型更擅长数学。
- 有的模型更擅长长文本理解。
- 有的模型更便宜、更快，但能力稍弱。

所以它希望做成：

```text
用户问题
  ↓
一个小调度器判断“这个问题该交给谁”
  ↓
被选中的 worker 模型回答
  ↓
返回一个最终答案
```

仓库里的 Fugu/TRINITY 线就是这种低延迟调度器。

![Routing overview](../assets/01_routing.png)

---

## 2. Qwen3-0.6B 在这里不是回答者

这是最容易误解的地方。

在普通聊天场景里，Qwen3-0.6B 会这样工作：

```text
输入 prompt
  ↓
Prefill：读完整个输入
  ↓
Decode：一个 token 一个 token 生成回答
  ↓
输出文本
```

但在 OpenFugu 的路由器里，Qwen3-0.6B 只做前半段：

```text
输入 prompt
  ↓
Prefill：读完整个输入
  ↓
取一个 hidden state 向量
  ↓
不 decode，不让 Qwen 自己回答
```

也就是说：

```text
Qwen3-0.6B 的作用 = 读题 + 提供内部理解向量
worker 模型的作用 = 真正回答问题
```

对应代码在 [openfugu/mini.py](../openfugu/mini.py)：

```python
out = self.model.model(**ids)          # 只跑 backbone，不用 lm_head 生成文本
return out.last_hidden_state[0, HIDDEN_POS, :]
```

这里的 `HIDDEN_POS = -2`，所以它取的是倒数第二个输入 token 的 hidden state。

---

## 3. 什么是 prefill，什么是 hidden state

LLM 推理一般分两段：

| 阶段 | 做什么 | OpenFugu 是否使用 |
| --- | --- | --- |
| Prefill | 把用户输入的全部 token 喂进模型，算出每个位置的内部状态 | 使用 |
| Decode | 根据前面的状态逐个生成回答 token | 不使用 Qwen 的 decode |

假设输入文本被分成这些 token：

```text
user, :, 证明, 勾股, 定理
```

模型完成 prefill 之后，每个 token 位置都会有一个向量：

```text
user  -> hidden state
:     -> hidden state
证明  -> hidden state
勾股  -> hidden state
定理  -> hidden state
```

Qwen3-0.6B 的 hidden size 是 `1024`，所以每个 hidden state 都是一个 1024 维向量：

```text
勾股 -> [0.12, -0.31, 0.08, ..., 1.44]   共 1024 个数
```

OpenFugu 取的是：

```python
last_hidden_state[0, -2, :]
```

含义是：

```text
第 0 条样本
倒数第 2 个输入 token 位置
全部 1024 个维度
```

如果 token 是：

```text
user, :, 证明, 勾股, 定理
```

那么 `-2` 就是 `勾股` 这个位置。

这里容易误会：`勾股 -> hidden state` 不是一本词典里“勾股”这个词的固定解释。

更好理解的方式是把模型想成一个边读边记笔记的人：

```text
读到 user       -> 笔记里知道：这是用户消息
读到 :          -> 笔记里知道：后面是用户内容
读到 证明       -> 笔记里知道：用户可能要一个推导/论证
读到 勾股       -> 笔记里知道：这是数学问题，和勾股定理有关
读到 定理       -> 笔记里知道：完整任务是“证明勾股定理”
```

每个位置的 hidden state 就像“读到这个位置时的笔记快照”。所以倒数第二个位置 `勾股` 的向量，更像是：

```text
“读到 user: 证明 勾股 时，模型脑子里的临时笔记”
```

它里面既有当前 token 的信息，也有前面上下文留下的痕迹。OpenFugu 正是拿这个“笔记快照”去判断：这个问题该交给哪个 worker 模型。

---

## 4. 从 hidden state 到“该叫哪个模型”

拿到 1024 维 hidden state 后，OpenFugu 做一件非常简单的事：

```python
logits = self.head @ h
```

这里：

- `h` 是 Qwen3-0.6B 的 hidden state，形状是 `(1024,)`。
- `self.head` 是一个线性头，形状是 `(10, 1024)`。
- 输出 `logits` 是 10 个分数。

这 10 个分数拆成两部分：

```text
前 7 个分数：选哪个 worker 模型
后 3 个分数：选什么角色
```

角色有三个：

| 角色 | 含义 |
| --- | --- |
| Worker | 直接解题或回答 |
| Thinker | 分析当前局面，给下一步建议 |
| Verifier | 检查当前答案是否可以接受 |

所以一次路由决策可以画成：

```text
用户问题
  ↓
Qwen3-0.6B prefill
  ↓
倒数第二个 token 的 1024 维 hidden state
  ↓
线性头 W_head @ h
  ↓
7 个 worker 分数 + 3 个角色分数
  ↓
选择 agent_id 和 role
```

---

## 5. `model_iter_60.npy` 里面装了什么

这个项目下载的关键文件是 `model_iter_60.npy`。它不是一个完整大模型，而是一个只有 `19456` 个浮点数的参数向量。

它分成两段：

```text
前 9216 个数：SVF offsets，用来轻量调整 Qwen3-0.6B 的部分权重
后 10240 个数：线性路由头，reshape 成 (10, 1024)
```

![Parameter vector](../assets/03_paramvec.png)

为什么是 `10240`？

```text
10 个输出分数 × 1024 维 hidden state = 10240
```

为什么是 `9216`？

它对 9 个矩阵做 SVF，每个矩阵贡献 1024 个奇异值偏移：

```text
9 × 1024 = 9216
```

---

## 6. SVF：它怎么轻量改 Qwen3-0.6B

SVF 是 Singular Value Fine-tuning，可以理解成“只调权重矩阵的奇异值强度”。

普通全参微调会改大量权重。SVF 不这么做。它对一个权重矩阵做 SVD：

```text
W = U · S · V^T
```

然后冻结 `U` 和 `V`，只调整中间的奇异值 `S`：

```text
S' = S × (1 + offset)
W' = U · S' · V^T
```

![SVF](../assets/02_svf.png)

这意味着：

- 它没有重写整个 Qwen3-0.6B。
- 它只是轻微改变一部分矩阵的“频谱强度”。
- 目标是让 Qwen 的 hidden state 更适合做“路由判断”。

在这个仓库验证出的 academic TRINITY checkpoint 中，SVF 涉及 9 个矩阵：

```text
embed_tokens
layer 26 的 q_proj
layer 26 的 k_proj
layer 26 的 v_proj
layer 26 的 o_proj
layer 26 的 gate_proj
layer 26 的 up_proj
layer 26 的 down_proj
lm_head
```

所以 OpenFugu 对 Qwen3-0.6B 的改造可以概括成：

```text
轻量调整部分权重表示能力 + 加一个线性路由头
```

---

## 7. 一次完整请求怎么跑

以用户问“证明勾股定理”为例，OpenFugu/TRINITY 的一次运行可以理解为：

```text
1. 把对话整理成 raw transcript
   例如：
   system: ...
   user: 证明勾股定理

2. 送入 Qwen3-0.6B
   只做 prefill，不生成回答

3. 取倒数第二个输入 token 的 hidden state
   得到一个 1024 维向量

4. 线性头打分
   得到 7 个 worker 分数和 3 个 role 分数

5. 选择某个 worker + 某个角色
   例如：
   agent_id = 2
   role = Worker

6. 把问题发给第 2 个 worker 模型
   worker 可能是 GPT、Claude、Gemini、DeepSeek、Gemma、Qwen-32B 等

7. worker 返回回答

8. 如果后续选到 Verifier
   Verifier 检查答案，ACCEPT 就结束，否则继续多轮
```

代码里的 coordinator 循环对应 [openfugu/mini.py](../openfugu/mini.py) 中的 `Coordinator.run()`。

---

## 8. 它是怎么训练出来的

这个项目里的思想是：不要手写规则说“数学找谁、代码找谁”，而是让系统通过任务结果学会路由。

训练时大概是：

```text
给一批问题
  ↓
尝试不同路由参数
  ↓
调用 worker 模型回答
  ↓
根据最终答案是否正确打分
  ↓
更新那 19456 个路由参数
```

这里用了 CMA-ES / sep-CMA-ES 一类梯度-free 优化方法。原因是 worker 模型可能是外部 API，最终奖励也可能只是“答对/答错”，不容易像普通神经网络那样反向传播梯度。

![sep-CMA-ES](../assets/04_sepcma.png)

可以把它理解成一种进化式搜索：

```text
当前参数
  ↓
随机生成一批候选参数
  ↓
每个候选参数跑任务，看谁效果好
  ↓
把好候选的方向合成下一代参数
  ↓
重复
```

最终保存出来的一个 checkpoint 就是：

```text
model_iter_60.npy
```

### 8.1 如果换成 LiveClawBench 训练

前面讲的是一般训练思路。你提到的 [Mosi-AI/LiveClawBench](https://github.com/Mosi-AI/LiveClawBench) 更接近真实 agent 任务，不是普通的“题目 + 标准答案”数据集。

它里面的任务更像：

```text
打开网页买东西
修复一个项目构建问题
整理资料并写入文件
处理邮件/日历/财务/社交媒体任务
在多个服务之间协调操作
```

这类任务很适合训练 router，因为不同 worker 模型的差异会更明显：

```text
有的模型擅长浏览器任务
有的模型擅长代码修复
有的模型擅长长上下文资料整理
有的模型擅长工具调用和步骤规划
```

LiveClawBench 的评分方式也和 GSM8K 不一样。GSM8K 可以直接比较数字答案；LiveClawBench 要启动一个真实/模拟环境，让 agent 完成任务，再由 verifier 给分。

所以接入方式是：

```text
LiveClawBench task instruction
  ↓
Qwen3-0.6B 做 prefill，取 hidden state
  ↓
router head 选择一个 worker
  ↓
用 LiveClawBench 的 Harbor CLI 跑这个 worker
  ↓
Harbor 执行任务、运行 verifier
  ↓
读取 /logs/verifier/reward.txt
  ↓
这个 reward 用来训练 router
```

也就是说，LiveClawBench 提供的不是简单标签，而是一个“真实评测环境 + 标准化分数”。

仓库里新增的入口是：

```text
train/train_trinity_liveclawbench.py
```

使用前需要先单独准备 LiveClawBench：

```bash
git clone https://github.com/Mosi-AI/LiveClawBench.git
cd LiveClawBench
./setup.sh
harbor --version
```

然后回到 OpenFugu 运行：

```bash
python train/train_trinity_liveclawbench.py \
  --liveclawbench-dir /path/to/LiveClawBench \
  --router-model Qwen/Qwen3-0.6B \
  --slot-models "custom/model-a,custom/model-b" \
  --ae CUSTOM_BASE_URL="$CUSTOM_BASE_URL" \
  --ae CUSTOM_API_KEY="$CUSTOM_API_KEY" \
  --n-train 8 \
  --iters 12 \
  --precompute-all
```

这条命令的意思是：

| 参数 | 含义 |
| --- | --- |
| `--liveclawbench-dir` | LiveClawBench 仓库路径 |
| `--router-model` | 用哪个 Qwen3-0.6B 作为 router backbone |
| `--slot-models` | 候选 worker 模型列表，由 router 在里面选择 |
| `--ae` | 传给 Harbor/OpenClaw agent 的环境变量，例如 API 地址和 key |
| `--n-train` | 选多少个 LiveClawBench 任务参与训练 |
| `--iters` | CMA-ES 训练迭代次数 |
| `--precompute-all` | 先跑完“每个任务 × 每个 worker”的真实分数，再训练 router |

为什么建议先用 `--precompute-all`？

因为 LiveClawBench 任务很重。每次真实评测都可能要启动 Docker、跑浏览器、调用模型、执行 verifier。如果每个 CMA-ES 候选参数都重新跑环境，成本会爆炸。

所以脚本采用了缓存矩阵：

```text
             worker A   worker B   worker C
task 1         1.0        0.5        0.0
task 2         0.0        1.0        0.5
task 3         0.5        0.0        1.0
```

训练时 router 不再重复跑 Docker，而是学习：

```text
看到 task 1 的 hidden state，应该选 worker A
看到 task 2 的 hidden state，应该选 worker B
看到 task 3 的 hidden state，应该选 worker C
```

这样成本上限变成：

```text
任务数 × worker 数
```

例如：

```text
8 个任务 × 2 个 worker = 16 次 Harbor 真实评测
```

之后 CMA-ES 训练就只是在缓存分数上搜索路由头，速度会快很多。

### 8.2 Colab 一键脚本

为了方便在 Colab 上测试，仓库里提供了一个脚本：

```text
scripts/colab_liveclawbench_router.sh
```

推荐做法是先复制一份配置文件，把多个 worker 都写进去：

```bash
!git clone https://github.com/luckfu/OpenFugu.git
%cd OpenFugu

!cp configs/liveclawbench_colab.example.yaml configs/my_workers.yaml
```

然后编辑：

```bash
configs/my_workers.yaml
```

配置文件里可以写多个 worker：

```yaml
workers:
  - name: gpt
    model: openai/gpt-4o

  - name: claude
    model: anthropic/claude-sonnet-4-5

  - name: custom_a
    model: custom/model-a
    ae:
      CUSTOM_BASE_URL: https://a.example.com/v1
      CUSTOM_API_KEY: key-a

  - name: custom_b
    model: custom/model-b
    ae:
      CUSTOM_BASE_URL: https://b.example.com/v1
      CUSTOM_API_KEY: key-b
```

这样每个 slot 的 endpoint 和 key 都能单独配置，不需要把所有 worker 硬塞进同一个 `CUSTOM_BASE_URL/CUSTOM_API_KEY`。

编辑好以后，一把跑起训练：

```bash
!CONFIG_FILE=configs/my_workers.yaml bash scripts/colab_liveclawbench_router.sh
```

这里的 slot 顺序就是 YAML 里的 `workers` 顺序：

```text
slot 0 -> custom/model-a -> https://a.example.com/v1 + key-a
slot 1 -> custom/model-b -> https://b.example.com/v1 + key-b
```

这个脚本会做几件事：

```text
安装 OpenFugu Python 依赖
下载 model_iter_60.npy 和 Qwen3-0.6B
克隆 LiveClawBench
运行 LiveClawBench 的 ./setup.sh
检查 Harbor / Docker
调用 train/train_trinity_liveclawbench.py 开始训练
```

几个常用环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `CONFIG_FILE` | 空 | 推荐入口，指定 YAML/JSON 配置文件 |
| `SLOT_MODELS` | `custom/model-a,custom/model-b` | worker slot 列表 |
| `CUSTOM_BASE_URL` | 空 | 全局 OpenAI-compatible worker API 地址 |
| `CUSTOM_API_KEY` | 空 | 全局 worker API key |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` 等 | 空 | 原生 provider worker 的 API key |
| `WORKER_AE` | 空 | 按 slot 传 Harbor agent env，格式是 `0:KEY=VALUE;1:KEY=VALUE` |
| `N_TRAIN` | `8` | 训练用 LiveClawBench 任务数 |
| `ITERS` | `12` | CMA-ES 迭代次数 |
| `DEVICE` | `cuda:0` | Qwen3-0.6B router backbone 运行设备 |
| `DOMAINS` | 空 | 可选，按 LiveClawBench domain 过滤 |
| `DIFFICULTIES` | 空 | 可选，例如 `easy|medium` |
| `INCLUDE_REGEX` | 空 | 可选，按任务目录名正则过滤 |

输出文件默认在 Colab 的 `/content` 下：

```text
/content/trinity_liveclawbench.npy      # 训练出的 router head
/content/liveclawbench_scores.csv       # task × worker 分数矩阵
/content/openfugu_liveclawbench_jobs    # Harbor job 和缓存
```

注意：LiveClawBench 的真实评测依赖 Docker。如果当前 Colab runtime 没有可用 Docker daemon，脚本会提前停止并提示。这个限制来自 LiveClawBench/Harbor，不是 OpenFugu 训练脚本本身。

### 8.3 几类训练数据的区别

现在可以把几种训练数据放在一起比较：

| 数据来源 | 任务形态 | reward 怎么来 | 适合做什么 |
| --- | --- | --- | --- |
| Mock | 人造 domain + 人造 worker 能力 | 脚本内部模拟 | 验证训练循环 |
| GSM8K | 数学题 | 数字答案匹配 | 最小真实训练闭环 |
| ToolScale | 工具调用规划 | 工具调用序列匹配 | 训练工具规划/多领域路由 |
| LiveClawBench | 真实 agent 任务 | Harbor verifier 的 `reward.txt` | 训练真实工作流场景下的 worker 选择 |

其中 LiveClawBench 更接近你想要的“复杂任务训练”。它的问题不只是“答对一道题”，而是“在环境里完成一件事”。

---

## 9. Fugu 和 Fugu-Ultra 的区别

仓库里还有 `openfugu/ultra.py`，对应另一条 Conductor/Fugu-Ultra 路线。

普通 Fugu/TRINITY 是：

```text
每一轮选择一个 worker
```

Fugu-Ultra/Conductor 是：

```text
先规划一个多步骤工作流
再按工作流调用多个 worker
```

Conductor 输出类似三组列表：

```text
model_id    = [2, 0, 1]
subtasks    = ["先分析问题", "写出解法", "检查答案"]
access_list = [[], [0], [0, 1]]
```

含义是：

```text
第 0 步：让模型 2 做分析
第 1 步：让模型 0 看第 0 步结果，写答案
第 2 步：让模型 1 看第 0、1 步结果，检查答案
```

![GRPO](../assets/05_grpo.png)

所以：

| 版本 | 调度方式 | 特点 |
| --- | --- | --- |
| Fugu / TRINITY | 每轮用 Qwen3-0.6B hidden state 选一个 worker | 快 |
| Fugu-Ultra / Conductor | 用一个更大的 Conductor 规划整个 workflow | 慢但更复杂 |

---

## 10. 最重要的误区

### 误区一：Qwen3-0.6B 自己在回答

不是。Qwen3-0.6B 在这里主要负责路由，不负责最终回答。

### 误区二：hidden state 是某个词的普通词向量

不完全是。它是模型读过上下文之后，在某个 token 位置产生的上下文状态。比如取到“勾股”位置的 hidden state，它不只是“勾股”这个词本身，而是“读到这里为止的上下文理解”。

### 误区三：还没开始 prefill 就能拿 hidden state

不是。hidden state 是 prefill 的产物。OpenFugu 是做完 prefill 后取 hidden state，然后跳过 Qwen 自己的 decode。

### 误区四：这是多个模型权重融合

不是。worker 模型权重没有被合并，也没有被改。OpenFugu 学的是“什么时候该叫谁”。

---

## 11. 一句话复盘

OpenFugu 做的是：

```text
用 Qwen3-0.6B 读问题
  ↓
拿 prefill 后的 hidden state
  ↓
用一个小线性头判断该调用哪个 worker 模型
  ↓
让 worker 真正回答
```

所以它把 Qwen3-0.6B 变成了一个低成本的 learned router。它的能力不来自 Qwen3-0.6B 自己变强了，而来自“会把合适的问题交给合适的模型”。
