# OpenFugu 统一评估器（EvalScope）运行文档

本文档说明如何使用 EvalScope 作为统一评估入口，对多个 worker 跑任意基准（BFCL、多轮 agent、通用推理），并把逐样本结果转换成 OpenFugu router 的训练矩阵。

与 `docs/BFCL_RUNBOOK.md` 的分工：

```text
BFCL router 正式训练数据   -> 继续用 eval_bfcl.py（断点续跑、失败重试更完善）
新能力面 / worker 快速摸底 -> 用本文档的 EvalScope 流程
```

两条路线的裁判是同一个：EvalScope 的 BFCL 评测底层也是官方 `bfcl_eval` 包。

不想敲命令行的话，第 6~9 步有对应的 Web 界面（见第 11 节）：

```bash
python openfugu/eval_ui.py --config configs/bfcl.yaml --port 8090
# 浏览器打开 http://127.0.0.1:8090
```

## 1. 这套流程评测什么

每个 worker 各自完整跑一遍相同的基准，EvalScope 负责下载数据集、调用模型、官方打分；OpenFugu 只保留一个转换层，把逐样本结果拼成 case × worker 矩阵：

```text
worker A -> evalscope eval -> reviews/*.jsonl ┐
worker B -> evalscope eval -> reviews/*.jsonl ├-> evalscope_to_matrix.py
worker C -> evalscope eval -> reviews/*.jsonl ┘      -> predictions.jsonl（喂训练器）
                                                      -> worker_matrix.csv（人工查看）
```

已实测的对齐性保证：不同 worker 独立运行同一 `--datasets --limit`，抽到的
`sample_id`、题目、`target` 完全一致，矩阵可以直接按 case_id 对齐。

## 2. 环境要求

```text
Python 3.10+
不需要 GPU
不需要 torch
需要 worker API key
磁盘约 1GB（.venv-eval 独立虚拟环境）
```

评估器安装在独立虚拟环境 `.venv-eval/`，不污染主环境（`bfcl-eval` 钉死
`tree_sitter==0.21.3`，直接装进主环境会连带降级一批包）。

## 3. 一次性搭建环境

```bash
bash scripts/setup_eval_env.sh
```

脚本会创建 `.venv-eval/` 并安装钉版依赖组合，结尾自检打印：

```text
evalscope=1.9.1 bfcl_eval 评分链路 import OK
```

脚本可重入，换机器或上云服务器直接重跑即可。所有版本钉死的原因写在脚本注释里，
不要单独升级其中某个包。

## 4. 配置 API key

```bash
export DEEPSEEK_API_KEY='...'
export ZHIPU_API_KEY='...'
```

不要把 key 写进 YAML、脚本或 shell 配置文件。

## 5. 查看可用基准

```bash
.venv-eval/bin/evalscope benchmark-info --list
.venv-eval/bin/evalscope benchmark-info bfcl_v4
```

与 OpenFugu 相关的常用基准：

```text
工具调用    bfcl_v4（22 子集，含 multi_turn/memory/web_search）、general_fc（自定义数据集）
多轮 agent  gaia、acebench、mcp_atlas、tau_bench
通用推理    gsm8k、aime24/25/26、math_500、mmlu 系、ifeval
```

注意 `memory_*` 子集依赖 faiss-cpu，macOS 14 以下没有轮子，跳过这些子集即可。

## 6. 冒烟测试

先用每类 3~5 条验证 API、评分器和输出文件正常，不适合比较模型能力：

```bash
.venv-eval/bin/evalscope eval \
  --model deepseek-v4-flash \
  --api-url https://api.deepseek.com/v1 \
  --api-key "$DEEPSEEK_API_KEY" \
  --eval-type openai_api \
  --datasets bfcl_v4 \
  --dataset-args '{"bfcl_v4":{"subset_list":["simple_python"]}}' \
  --limit 3 \
  --work-dir openfugu_evalscope/deepseek_v4_flash
```

输出目录结构：

```text
openfugu_evalscope/deepseek_v4_flash/<时间戳>/
  reviews/<模型>/<数据集>_<子集>.jsonl   逐样本记录（转换器的输入）
  reports/                              汇总报告与 HTML
  predictions/                          模型原始输出
```

## 7. 正式评测

每个 worker 单独跑一遍，`--datasets`、`--dataset-args`、`--limit` 必须完全一致，
`--work-dir` 每个 worker 用独立目录：

```bash
# worker 1
.venv-eval/bin/evalscope eval \
  --model deepseek-v4-flash \
  --api-url https://api.deepseek.com/v1 \
  --api-key "$DEEPSEEK_API_KEY" \
  --eval-type openai_api \
  --datasets bfcl_v4 \
  --dataset-args '{"bfcl_v4":{"subset_list":["simple_python","parallel","multiple","parallel_multiple"]}}' \
  --generation-config timeout=600 \
  --work-dir openfugu_evalscope/deepseek_v4_flash

# worker 2
.venv-eval/bin/evalscope eval \
  --model deepseek-v4-pro \
  --api-url https://api.deepseek.com/v1 \
  --api-key "$DEEPSEEK_API_KEY" \
  --eval-type openai_api \
  --datasets bfcl_v4 \
  --dataset-args '{"bfcl_v4":{"subset_list":["simple_python","parallel","multiple","parallel_multiple"]}}' \
  --generation-config timeout=600 \
  --work-dir openfugu_evalscope/deepseek_v4_pro

# worker 3
.venv-eval/bin/evalscope eval \
  --model glm-5.2 \
  --api-url https://open.bigmodel.cn/api/coding/paas/v4 \
  --api-key "$ZHIPU_API_KEY" \
  --eval-type openai_api \
  --datasets bfcl_v4 \
  --dataset-args '{"bfcl_v4":{"subset_list":["simple_python","parallel","multiple","parallel_multiple"]}}' \
  --generation-config timeout=600 \
  --work-dir openfugu_evalscope/glm_5_2
```

`--generation-config timeout=600` 必须加。实测默认超时下智谱 glm-5.2 全部超时，
EvalScope 会把超时静默记成 0 分（BFCL 场景下还伪装成
`ast_decoder:decoder_failed`），只看汇总表会得出完全错误的能力结论。

不加 `--limit` 即全量。跑其他基准把 `--datasets` 换掉即可，例如：

```bash
--datasets gsm8k
--datasets gaia
```

## 8. 转换成训练矩阵

```bash
python eval/evalscope_to_matrix.py \
  --run DeepSeek-V4-Flash=openfugu_evalscope/deepseek_v4_flash \
  --run DeepSeek-V4-Pro=openfugu_evalscope/deepseek_v4_pro \
  --run zhipu_glm_5_2=openfugu_evalscope/glm_5_2 \
  --out-predictions openfugu_bfcl/evalscope_predictions.jsonl \
  --out-matrix openfugu_bfcl/evalscope_worker_matrix.csv
```

要点：

```text
--run 名称   必须与 configs/*.yaml 中 workers 的 name 一致，顺序即矩阵列顺序
--run 目录   传 --work-dir 即可，自动取其中最新时间戳目录
--metric     多指标基准（如 ifeval）需要手动指定主指标
```

转换器会做失败分类。超时、限流、欠费、鉴权失败属于执行失败，不是能力问题：

```text
--on-failure exclude（默认） 整条 case 从矩阵剔除，并在 stderr 列出待重跑清单
--on-failure zero            按 0 分计入（仅当你明确要把服务不可用算作失败时使用）
```

典型输出：

```text
[evalscope-matrix] worker=['DeepSeek-V4-Flash', 'DeepSeek-V4-Pro', 'zhipu_glm_5_2']
[evalscope-matrix] 可用 case=198 剔除(执行失败)=2
[evalscope-matrix] DeepSeek-V4-Flash: 176/198 = 88.89%
被剔除的 case（对失败 worker 重跑 EvalScope 后重新转换）:
  simple_python_0 <- zhipu_glm_5_2: Request timed out.
```

有剔除时的处理：对失败的 worker 用相同参数重跑第 7 步（EvalScope 会重新调用），
然后重新执行本步转换。

## 9. 训练 router

转换产物与 `eval_bfcl.py` 的输出同构，训练器一行不改：

```bash
python train/train_trinity_bfcl.py \
  --config configs/bfcl.yaml \
  --predictions openfugu_bfcl/evalscope_predictions.jsonl \
  --matrix-out openfugu_bfcl/evalscope_worker_matrix.csv \
  --out openfugu_bfcl/trinity_evalscope.npy
```

训练阶段需要加载 `Qwen/Qwen3-0.6B`，要求与 `docs/BFCL_RUNBOOK.md` 第 8 节相同。

## 10. 常见坑

```text
pip install 'evalscope[bfcl]'   不要用。extra 解析会把 evalscope 回退到 1.5.x，
                                bfcl_eval 还装不上。用 setup_eval_env.sh。

汇总分数异常低                   先翻 reviews/*.jsonl 看 prediction 字段。
                                执行失败（Request timed out 等）会被记成 0 分，
                                不代表模型能力。加大 timeout 重跑。

不同 worker case 数不一致        各 run 的 --datasets/--dataset-args/--limit
                                必须完全一致，转换器只保留交集并给出警告。

memory_* 子集报 faiss 错误       macOS 14 以下无 faiss-cpu 轮子，跳过这些子集。

EvalScope 没有失败重试           对失败 worker 整体重跑即可（幂等，结果目录
                                按时间戳分开，转换器自动取最新一次）。
```

## 11. Web 界面（评估台）

命令行三步在浏览器里点按钮完成：

```bash
export DEEPSEEK_API_KEY='...' ZHIPU_API_KEY='...'
python openfugu/eval_ui.py --config configs/bfcl.yaml --port 8090
```

打开 `http://127.0.0.1:8090`：

```text
区块 1  勾选 worker、选基准和子集、填条数与超时 -> 发起评测；
        “⚙ 模型配置”可在界面上增删改 worker（写回 configs/bfcl.yaml 的
        workers 段，其余配置不动）、发真实请求测连通、临时注入 API key；
        基准旁的“预览”按钮弹窗展示测试集样题（BFCL 读仓库内置题库，
        其他基准读 modelscope 本地缓存；没下过的基准弹窗里给“下载
        数据集”按钮，后台下好自动展示样题——先预览、再发起评测）
区块 2  各 worker 实时进度条、耗时、通过率；执行失败（超时/限流）单独
        标黄提示，一键“重试失败条目”只补跑失败样本，成功结果复用缓存
区块 3  一键调用 evalscope_to_matrix.py 合并矩阵，页面直接显示
        case × worker 表格和下一步训练命令
```

失败重试的实现：EvalScope 的 --use-cache 不分成败一律复用缓存，所以评估台
会先把执行失败的样本从 predictions/reviews 两级缓存里剪掉，再带 --use-cache
续跑，EvalScope 就只补跑这几条。重试参数与原跑完全一致（记在
 openfugu_evalscope/<worker>/last_run.json），不会破坏样本对齐。

说明：

```text
服务只绑 127.0.0.1，不要开放到公网（它能读 API key、起子进程）
评测子进程仍跑在 .venv-eval，本服务只做编排，主环境无新增依赖
评测输出在 openfugu_evalscope/<worker>/，日志里 API key 已打码
界面上的失败分类与转换器同一套代码（复用 load_run），口径一致
界面上填的 key 值只存进程内存不落盘，重启服务后需重新注入或 export
```
