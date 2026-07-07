const pptxgen = require("/tmp/openfugu-pptx-node/node_modules/pptxgenjs");
const path = require("path");

const pptx = new pptxgen();
pptx.defineLayout({ name: "OPENFUGU_16_9", width: 10, height: 5.625 });
pptx.layout = "OPENFUGU_16_9";
pptx.author = "OpenFugu";
pptx.subject = "OpenFugu principles, training, and cost";
pptx.title = "OpenFugu 原理、性能优势与训练成本";
pptx.company = "OpenFugu";
pptx.lang = "zh-CN";

const S = pptx.ShapeType;
const C = {
  ink: "0A0A0A",
  slate: "264653",
  teal: "2A9D8F",
  gold: "E9C46A",
  orange: "F4A261",
  red: "E76F51",
  blue: "0070F3",
  navy: "023047",
  paper: "FFFFFF",
  soft: "F7F7F7",
  line: "D4D4D4",
  muted: "525252",
  paleBlue: "E0EFFF",
  paleGold: "FEF9E7",
};

const W = 10;
const H = 5.625;
const font = "Microsoft YaHei";
const root = path.resolve(__dirname, "..");
const img = (name) => path.join(root, "assets", name);
const imgMeta = {
  "01_routing.png": { w: 836, h: 102 },
  "02_svf.png": { w: 719, h: 88 },
  "03_paramvec.png": { w: 651, h: 89 },
  "04_sepcma.png": { w: 890, h: 44 },
  "05_grpo.png": { w: 662, h: 87 },
};

function addImageContain(slide, name, x, y, w, h, opts = {}) {
  const meta = imgMeta[name];
  const ratio = meta.w / meta.h;
  let iw = w;
  let ih = w / ratio;
  if (ih > h) {
    ih = h;
    iw = h * ratio;
  }
  const ix = x + (w - iw) / 2;
  const iy = y + (h - ih) / 2;
  if (opts.frame) {
    slide.addShape(S.roundRect, {
      x, y, w, h, rectRadius: 0.06,
      fill: { color: opts.fill || C.soft },
      line: { color: opts.line || C.line, transparency: 10 },
    });
  }
  slide.addImage({ path: img(name), x: ix, y: iy, w: iw, h: ih });
  return { x: ix, y: iy, w: iw, h: ih };
}

function slideBase(slide, idx, title, kicker) {
  slide.background = { color: C.paper };
  slide.addShape(S.rect, { x: 0, y: 0, w: W, h: H, fill: { color: C.paper }, line: { color: C.paper } });
  slide.addText(kicker || "OpenFugu", {
    x: 0.45, y: 0.25, w: 2.0, h: 0.25,
    fontFace: font, fontSize: 8.5, color: C.teal, bold: true, margin: 0,
  });
  slide.addText(title, {
    x: 0.45, y: 0.55, w: 8.9, h: 0.45,
    fontFace: font, fontSize: 24, color: C.ink, bold: true, margin: 0,
    fit: "shrink",
  });
  if (idx) addBadge(slide, idx);
}

function addBadge(slide, idx) {
  slide.addShape(S.ellipse, { x: 9.32, y: 5.1, w: 0.34, h: 0.34, fill: { color: C.slate }, line: { color: C.slate } });
  slide.addText(String(idx), {
    x: 9.32, y: 5.105, w: 0.34, h: 0.32,
    fontFace: "Arial", fontSize: 9.5, color: C.paper, bold: true, align: "center", valign: "mid", margin: 0,
  });
}

function addPill(slide, text, x, y, w, color, textColor = C.paper) {
  slide.addShape(S.roundRect, { x, y, w, h: 0.28, rectRadius: 0.08, fill: { color }, line: { color } });
  slide.addText(text, { x: x + 0.08, y: y + 0.06, w: w - 0.16, h: 0.14, fontFace: font, fontSize: 8.2, color: textColor, bold: true, margin: 0, fit: "shrink" });
}

function addCard(slide, x, y, w, h, title, body, accent = C.teal) {
  slide.addShape(S.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: C.soft }, line: { color: C.line, transparency: 10 } });
  slide.addShape(S.rect, { x, y, w: 0.08, h, fill: { color: accent }, line: { color: accent } });
  slide.addText(title, { x: x + 0.22, y: y + 0.18, w: w - 0.34, h: 0.22, fontFace: font, fontSize: 12.5, color: C.ink, bold: true, margin: 0, fit: "shrink" });
  slide.addText(body, { x: x + 0.22, y: y + 0.55, w: w - 0.34, h: h - 0.68, fontFace: font, fontSize: 9.2, color: C.muted, breakLine: false, margin: 0.02, fit: "shrink", valign: "top" });
}

function addFlowNode(slide, x, y, w, h, title, body, color) {
  slide.addShape(S.roundRect, { x, y, w, h, rectRadius: 0.07, fill: { color }, line: { color } });
  slide.addText(title, { x: x + 0.12, y: y + 0.14, w: w - 0.24, h: 0.22, fontFace: font, fontSize: 10.5, color: C.paper, bold: true, margin: 0, align: "center", fit: "shrink" });
  slide.addText(body, { x: x + 0.12, y: y + 0.45, w: w - 0.24, h: h - 0.54, fontFace: font, fontSize: 7.6, color: C.paper, margin: 0.01, align: "center", fit: "shrink" });
}

function addArrow(slide, x1, y1, x2, y2, color = C.slate) {
  slide.addShape(S.line, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color, width: 1.6, beginArrowType: "none", endArrowType: "triangle" } });
}

function bulletList(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  for (const item of items) {
    runs.push({ text: item, options: { bullet: { type: "bullet" }, hanging: 3 } });
  }
  slide.addText(runs, {
    x, y, w, h, fontFace: font, fontSize: opts.size || 11,
    color: opts.color || C.ink, breakLine: true, margin: 0.02,
    paraSpaceAfterPt: opts.space || 5, fit: "shrink",
  });
}

function addBar(slide, label, value, x, y, w, color, suffix = "") {
  slide.addText(label, { x, y, w: 1.7, h: 0.16, fontFace: font, fontSize: 8.2, color: C.muted, margin: 0, fit: "shrink" });
  slide.addShape(S.rect, { x: x + 1.9, y: y + 0.02, w, h: 0.13, fill: { color: "EDEDED" }, line: { color: "EDEDED" } });
  slide.addShape(S.rect, { x: x + 1.9, y: y + 0.02, w: w * value, h: 0.13, fill: { color }, line: { color } });
  slide.addText(`${Math.round(value * 100)}${suffix}`, { x: x + 1.9 + w + 0.12, y: y - 0.01, w: 0.55, h: 0.18, fontFace: "Arial", fontSize: 8, color, bold: true, margin: 0 });
}

// 01 Cover
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  slide.addShape(S.rect, { x: 0, y: 0, w: W, h: H, fill: { color: C.navy }, line: { color: C.navy } });
  slide.addText("OpenFugu", { x: 0.62, y: 1.12, w: 4.5, h: 0.78, fontFace: "Arial", fontSize: 44, color: C.paper, bold: true, margin: 0 });
  slide.addText("原理、性能优势与训练成本", { x: 0.62, y: 1.96, w: 5.2, h: 0.42, fontFace: font, fontSize: 22, color: C.gold, bold: true, margin: 0 });
  slide.addText("把 Qwen3-0.6B 从回答者变成 learned router，让合适的问题交给合适的 worker 模型。", { x: 0.66, y: 2.58, w: 7.5, h: 0.42, fontFace: font, fontSize: 13.5, color: "F7F7F7", margin: 0.02, fit: "shrink" });
  addPill(slide, "TRINITY / Fugu line", 0.66, 3.25, 1.45, C.teal);
  addPill(slide, "SVF + Linear Head", 2.25, 3.25, 1.55, C.gold, C.ink);
  addPill(slide, "LiveClawBench", 3.95, 3.25, 1.25, C.red);
  addImageContain(slide, "01_routing.png", 0.68, 3.85, 8.65, 1.08, { frame: true, fill: "FFFFFF" });
  slide.addText("2026 · 技术说明", { x: 0.66, y: 4.92, w: 2.8, h: 0.18, fontFace: font, fontSize: 9, color: C.line, margin: 0 });
}

// 02 TOC
{
  const slide = pptx.addSlide();
  slideBase(slide, 2, "目录：从问题到训练落地", "Overview");
  const sections = [
    ["01", "解决什么问题", "多模型能力差异、成本与供应商风险"],
    ["02", "核心原理", "Prefill hidden state → 路由头 → worker"],
    ["03", "训练方法", "任务 × worker reward 矩阵与 CMA-ES"],
    ["04", "性能与成本", "何时省钱、何时更强、何时不划算"],
    ["05", "落地路径", "slot 配置、LiveClawBench 在线/离线训练"],
  ];
  sections.forEach((s, i) => {
    const y = 1.28 + i * 0.68;
    slide.addText(s[0], { x: 0.75, y, w: 0.58, h: 0.34, fontFace: "Arial", fontSize: 18, color: C.teal, bold: true, margin: 0 });
    slide.addText(s[1], { x: 1.45, y: y + 0.02, w: 2.8, h: 0.24, fontFace: font, fontSize: 13, color: C.ink, bold: true, margin: 0 });
    slide.addText(s[2], { x: 4.15, y: y + 0.04, w: 4.6, h: 0.22, fontFace: font, fontSize: 10.2, color: C.muted, margin: 0 });
    slide.addShape(S.line, { x: 0.75, y: y + 0.46, w: 8.25, h: 0, line: { color: C.line, transparency: 30, width: 0.8 } });
  });
}

// 03 Problem
{
  const slide = pptx.addSlide();
  slideBase(slide, 3, "要解决的问题：不是“再造一个大模型”", "Problem");
  addCard(slide, 0.55, 1.28, 2.75, 2.9, "单模型依赖", "一个模型很难在代码、数学、检索、工具调用、长上下文等所有场景同时最优；供应商可用性和政策也会变化。", C.red);
  addCard(slide, 3.62, 1.28, 2.75, 2.9, "成本不均衡", "很多任务不需要最强模型；简单任务全量调用昂贵模型，会把平均成本和延迟拉高。", C.gold);
  addCard(slide, 6.68, 1.28, 2.75, 2.9, "任务粒度不同", "真实 agent 工作流不是一道题，而是多步操作、校验和修复。需要按任务选择模型，而不是固定一个入口。", C.teal);
  slide.addText("目标：把多个 worker 模型组织成一个统一入口，同时让路由策略可训练、可替换、可评估。", { x: 0.62, y: 4.62, w: 8.75, h: 0.28, fontFace: font, fontSize: 13, color: C.slate, bold: true, align: "center", margin: 0 });
}

// 04 Architecture
{
  const slide = pptx.addSlide();
  slideBase(slide, 4, "OpenFugu 的答案：一个 learned router，而不是融合权重", "Architecture");
  addFlowNode(slide, 0.55, 1.55, 1.25, 0.9, "用户请求", "prompt / 对话", C.slate);
  addArrow(slide, 1.88, 2.0, 2.42, 2.0);
  addFlowNode(slide, 2.5, 1.38, 1.5, 1.25, "Qwen3-0.6B", "只做 prefill\n不生成答案", C.teal);
  addArrow(slide, 4.08, 2.0, 4.62, 2.0);
  addFlowNode(slide, 4.72, 1.38, 1.5, 1.25, "路由头", "W_head @ h\n选 slot/role", C.gold);
  addArrow(slide, 6.3, 2.0, 6.84, 2.0);
  addFlowNode(slide, 6.95, 0.92, 1.25, 0.72, "Worker 0", "便宜/快", C.blue);
  addFlowNode(slide, 6.95, 1.82, 1.25, 0.72, "Worker 1", "强通用", C.red);
  addFlowNode(slide, 6.95, 2.72, 1.25, 0.72, "Worker 2", "推理/代码", C.navy);
  addArrow(slide, 8.28, 2.0, 8.82, 2.0);
  addFlowNode(slide, 8.9, 1.55, 0.75, 0.9, "答案", "返回给用户", C.slate);
  addImageContain(slide, "01_routing.png", 0.75, 3.58, 8.55, 1.08, { frame: true, fill: "FFFFFF" });
  slide.addText("关键点：worker 权重不被修改，OpenFugu 学的是“什么时候叫谁”。", { x: 0.72, y: 4.95, w: 8.2, h: 0.22, fontFace: font, fontSize: 10.5, color: C.muted, margin: 0 });
}

// 05 Hidden State
{
  const slide = pptx.addSlide();
  slideBase(slide, 5, "Qwen3-0.6B 做了什么：只取 prefill 后的 hidden state", "Mechanism");
  addCard(slide, 0.55, 1.18, 2.75, 3.35, "普通聊天模型", "Prefill 读入 prompt，然后 Decode 一个 token 一个 token 生成回答。", C.slate);
  addCard(slide, 3.62, 1.18, 2.75, 3.35, "Fugu router", "只做 Prefill，取 last_hidden_state[0, -2, :]，跳过 Qwen 自己的 Decode。", C.teal);
  addCard(slide, 6.68, 1.18, 2.75, 3.35, "直觉理解", "这个向量像“读到倒数第二个 token 时的笔记快照”，包含当前 token 和前文上下文痕迹。", C.gold);
  slide.addText("token: user, :, 证明, 勾股, 定理", { x: 0.72, y: 4.78, w: 3.2, h: 0.18, fontFace: "Consolas", fontSize: 9.5, color: C.muted, margin: 0 });
  slide.addText("取 -2 → “勾股”位置的 1024 维上下文状态，而不是词典里的固定词向量。", { x: 4.05, y: 4.78, w: 5.0, h: 0.2, fontFace: font, fontSize: 10, color: C.slate, margin: 0 });
}

// 06 Param vector
{
  const slide = pptx.addSlide();
  slideBase(slide, 6, "参数改造：19,456 个数，而不是全量微调 0.6B", "Parameters");
  addImageContain(slide, "03_paramvec.png", 0.68, 1.12, 8.65, 1.18, { frame: true, fill: "FFFFFF" });
  addCard(slide, 0.72, 2.75, 2.75, 1.42, "SVF offsets：9,216", "9 个矩阵 × 1024 个奇异值偏移；轻量改变表示空间。", C.teal);
  addCard(slide, 3.68, 2.75, 2.75, 1.42, "Linear head：10,240", "(7 worker + 3 role) × 1024；把 hidden state 映射为路由分数。", C.gold);
  addCard(slide, 6.65, 2.75, 2.75, 1.42, "部署含义", "训练 head 可以很轻；换 worker 池时，通常先 head-only 校准。", C.red);
  slide.addText("总参数量约 19.5K，远低于 LoRA/全参微调；这也是可以用 CMA-ES 搜索的原因。", { x: 0.72, y: 4.95, w: 8.5, h: 0.22, fontFace: font, fontSize: 10.2, color: C.muted, margin: 0 });
}

// 07 SVF
{
  const slide = pptx.addSlide();
  slideBase(slide, 7, "SVF：只调整权重矩阵的奇异值强度", "SVF");
  addImageContain(slide, "02_svf.png", 0.68, 1.10, 8.65, 1.06, { frame: true, fill: "FFFFFF" });
  slide.addText("W = U · S · Vᵀ", { x: 0.95, y: 2.65, w: 3.6, h: 0.34, fontFace: "Cambria", fontSize: 22, color: C.slate, bold: true, margin: 0 });
  slide.addText("S' = S × (1 + offset)", { x: 0.95, y: 3.35, w: 3.8, h: 0.34, fontFace: "Cambria", fontSize: 22, color: C.teal, bold: true, margin: 0 });
  bulletList(slide, [
    "冻结 U/V，只学习奇异值缩放",
    "0 offset 等于原始 Qwen 权重",
    "能改变表示空间，但不会重写整个模型",
    "适合把 backbone 调成“路由特征提取器”",
  ], 5.05, 2.65, 4.0, 1.25, { size: 10.2, space: 4 });
}

// 08 Training
{
  const slide = pptx.addSlide();
  slideBase(slide, 8, "训练闭环：训练的是选择策略，不是 worker 本身", "Training");
  addFlowNode(slide, 0.62, 1.28, 1.3, 0.85, "任务集", "问题/环境", C.slate);
  addArrow(slide, 2.02, 1.7, 2.58, 1.7);
  addFlowNode(slide, 2.68, 1.28, 1.35, 0.85, "候选参数", "head/SVF", C.teal);
  addArrow(slide, 4.13, 1.7, 4.68, 1.7);
  addFlowNode(slide, 4.78, 1.28, 1.35, 0.85, "调用 worker", "回答/执行", C.gold);
  addArrow(slide, 6.23, 1.7, 6.78, 1.7);
  addFlowNode(slide, 6.88, 1.28, 1.35, 0.85, "Reward", "0~1 分数", C.red);
  addArrow(slide, 8.23, 1.7, 8.72, 1.7);
  addFlowNode(slide, 8.78, 1.28, 0.72, 0.85, "更新", "CMA", C.navy);
  addImageContain(slide, "04_sepcma.png", 0.72, 2.62, 8.55, 0.72, { frame: true, fill: "FFFFFF" });
  addCard(slide, 1.05, 3.62, 7.9, 1.1, "为什么不用普通反向传播？", "worker 可能是外部 API；reward 可能来自执行环境/verifier；信号稀疏且不可微。因此用 sep-CMA-ES 这类黑盒优化更自然。", C.teal);
}

// 09 LiveClawBench
{
  const slide = pptx.addSlide();
  slideBase(slide, 9, "训练数据：从题库到真实 agent benchmark", "LiveClawBench");
  addCard(slide, 0.55, 1.18, 2.95, 3.35, "GSM8K / MATH", "数字或公式答案可直接校验；成本低，但任务类型单一，worker 差异有限。", C.gold);
  addCard(slide, 3.72, 1.18, 2.95, 3.35, "ToolScale", "工具调用规划任务；可用 expected actions 做 reward，适合训练工具/计划能力路由。", C.teal);
  addCard(slide, 6.9, 1.18, 2.55, 3.35, "LiveClawBench", "真实 agent 任务：网页、邮件、代码、财务、知识库；reward 来自 Harbor verifier。", C.red);
  slide.addText("线上路径需要 Docker/Harbor；Colab 无 Docker 时，使用 HuggingFace 已发布 trajectories 做离线 warm start。", { x: 0.72, y: 4.78, w: 8.3, h: 0.28, fontFace: font, fontSize: 11, color: C.slate, bold: true, align: "center", margin: 0 });
}

// 10 Online vs Offline
{
  const slide = pptx.addSlide();
  slideBase(slide, 10, "LiveClawBench：在线评测与离线 trajectories 两条路", "Training paths");
  addCard(slide, 0.65, 1.25, 4.1, 3.25, "在线 Harbor 训练", "真实调用你的 worker 服务商，Docker 启动任务环境，verifier 写 reward.txt。\n\n优点：分数就是你当前模型池的真实表现。\n代价：慢、贵、需要 Docker。", C.teal);
  addCard(slide, 5.25, 1.25, 4.1, 3.25, "离线 trajectories 训练", "读取 Mosi-AI/LiveClawbench-trajectories，聚合 task × model 历史分数。\n\n优点：Colab 可跑、无 Docker。\n限制：只能覆盖已发布历史模型名。", C.gold);
  slide.addText("实践建议：先离线训练得到 warm start，再在有 Docker 的机器上用小批任务校准当前 worker 池。", { x: 0.78, y: 4.82, w: 8.3, h: 0.24, fontFace: font, fontSize: 10.5, color: C.muted, margin: 0, align: "center" });
}

// 11 Performance
{
  const slide = pptx.addSlide();
  slideBase(slide, 11, "性能优势：不是每次都更强，而是提升平均决策质量", "Performance");
  addBar(slide, "固定单模型", 0.48, 0.9, 1.38, 4.8, C.red, "%");
  addBar(slide, "随机/手写规则", 0.56, 0.9, 1.78, 4.8, C.gold, "%");
  addBar(slide, "训练后 router", 0.82, 0.9, 2.18, 4.8, C.teal, "%");
  addBar(slide, "Oracle 上限", 0.95, 0.9, 2.58, 4.8, C.slate, "%");
  addCard(slide, 0.65, 3.22, 2.75, 1.5, "质量收益", "当 worker 能力互补时，按任务路由可以超过任何固定单模型。仓库 eval 中训练 router 相对 best single 有明显提升。", C.teal);
  addCard(slide, 3.62, 3.22, 2.75, 1.5, "成本收益", "简单任务落到便宜模型，复杂任务才交给贵模型；平均成本低于总是调用最强模型。", C.gold);
  addCard(slide, 6.58, 3.22, 2.85, 1.5, "延迟收益", "TRINITY 路由只需一次 Qwen3-0.6B prefill，无需 router 自己 decode 完整回答。", C.slate);
  slide.addText("注意：如果 worker 能力接近、任务单一或 reward 噪声大，路由收益会变小。", { x: 0.8, y: 4.94, w: 8.0, h: 0.18, fontFace: font, fontSize: 9.5, color: C.muted, margin: 0 });
}

// 12 Cost model
{
  const slide = pptx.addSlide();
  slideBase(slide, 12, "训练成本：真正贵的是 task × worker 的 reward 获取", "Cost");
  const rows = [
    ["离线 trajectories", "无需 Docker/API", "下载 parquet + Qwen 特征提取", "Colab 可跑"],
    ["在线 Harbor", "N_tasks × N_workers", "每次要启动环境并调用模型", "最真实也最贵"],
    ["head-only 校准", "只训练 n_workers × 1024", "不动 SVF/Qwen 权重", "换 worker 池首选"],
    ["全量 SVF+head", "19,456 维", "需要更多候选与 reward", "用于重建原始 checkpoint"],
  ];
  slide.addShape(S.rect, { x: 0.6, y: 1.15, w: 8.8, h: 0.42, fill: { color: C.slate }, line: { color: C.slate } });
  ["路径", "主要成本", "瓶颈", "建议"].forEach((h, i) => {
    slide.addText(h, { x: [0.75, 2.45, 4.78, 7.3][i], y: 1.28, w: [1.3, 2, 2.1, 1.5][i], h: 0.15, fontFace: font, fontSize: 8.5, color: C.paper, bold: true, margin: 0 });
  });
  rows.forEach((r, i) => {
    const y = 1.72 + i * 0.72;
    slide.addShape(S.rect, { x: 0.6, y: y - 0.1, w: 8.8, h: 0.52, fill: { color: i % 2 === 0 ? C.soft : C.paper }, line: { color: C.line, transparency: 35 } });
    [0.75, 2.45, 4.78, 7.3].forEach((x, j) => {
      slide.addText(r[j], { x, y, w: [1.45, 2.0, 2.2, 1.75][j], h: 0.22, fontFace: font, fontSize: 8.4, color: j === 0 ? C.ink : C.muted, bold: j === 0, margin: 0.01, fit: "shrink" });
    });
  });
  slide.addText("在线成本估算：8 个任务 × 3 个 worker = 24 次完整 Harbor 评测；离线训练则直接复用已发布分数矩阵。", { x: 0.7, y: 4.92, w: 8.2, h: 0.22, fontFace: font, fontSize: 10, color: C.slate, bold: true, margin: 0 });
}

// 13 Deployment
{
  const slide = pptx.addSlide();
  slideBase(slide, 13, "部署与配置：slot 是稳定契约", "Deployment");
  addCard(slide, 0.55, 1.15, 2.85, 3.4, "slot 顺序", "router 输出 agent_id。YAML workers 的顺序就是 slot 0/1/2。换顺序等于换语义。", C.teal);
  addCard(slide, 3.58, 1.15, 2.85, 3.4, "同分 tie-break", "argmax 并列时选择更小 slot。把便宜/快/够用的模型放前面，贵 reasoning 模型放后面。", C.gold);
  addCard(slide, 6.62, 1.15, 2.85, 3.4, "换 worker 后", "不要盲用旧 head。先跑离线/小批在线 score matrix，再做 head-only 训练。", C.red);
  slide.addText("配置建议：deepseek_chat → zhipu_glm_5_2 → deepseek_reasoner，表达“同分时优先低成本”。", { x: 0.72, y: 4.86, w: 8.5, h: 0.24, fontFace: font, fontSize: 10.5, color: C.slate, align: "center", margin: 0 });
}

// 14 Summary
{
  const slide = pptx.addSlide();
  slide.background = { color: C.slate };
  slide.addShape(S.rect, { x: 0, y: 0, w: W, h: H, fill: { color: C.slate }, line: { color: C.slate } });
  slide.addText("结论", { x: 0.65, y: 0.65, w: 2.2, h: 0.55, fontFace: font, fontSize: 34, color: C.paper, bold: true, margin: 0 });
  const points = [
    ["定位", "OpenFugu 是 learned router，不是权重融合模型。"],
    ["原理", "Qwen3-0.6B 只做 prefill，hidden state 经线性头选择 worker。"],
    ["训练", "核心数据是 task × worker reward；在线最真实，离线最适合 Colab warm start。"],
    ["成本", "训练贵在获取 reward；服务贵在 worker 调用，router 本身很轻。"],
    ["落地", "slot 顺序稳定，同分优先便宜/快模型，换 worker 后重训 head。"],
  ];
  points.forEach((p, i) => {
    const y = 1.55 + i * 0.58;
    slide.addText(p[0], { x: 0.85, y, w: 0.72, h: 0.2, fontFace: font, fontSize: 11, color: C.gold, bold: true, margin: 0 });
    slide.addText(p[1], { x: 1.72, y, w: 7.25, h: 0.2, fontFace: font, fontSize: 11.5, color: C.paper, margin: 0, fit: "shrink" });
  });
  slide.addText("下一步：在 Colab 跑 offline trajectories，拿到初始 head；再到有 Docker 的机器用 LiveClawBench 在线校准。", { x: 0.85, y: 4.92, w: 8.0, h: 0.24, fontFace: font, fontSize: 10.5, color: C.line, margin: 0 });
  addBadge(slide, 14);
}

pptx.writeFile({ fileName: path.join(__dirname, "output", "openfugu_principles_training_cost.pptx") });
