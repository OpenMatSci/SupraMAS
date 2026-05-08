# Forward Idea Benchmark 使用手册

本手册对应旧版 `forward_idea_benchmark.py` 论文抽取流程。当前主流程默认直接使用
`results/benchmark_output/Idea_forward/` 中的标签和评估结果；`data/` 下的论文 PDF 和
当前输入 JSON 作为 `paper_idea.json`、`label_strategies.json`、`label_experiment.json` 的来源与可追溯材料保留。

目标：评估 baseline 生成的 `idea.json` 在“前瞻性”上的表现，包含 3 个维度：
1. 策略重合度（`PolicyAlignment`）
2. 组合相似度（`CombinationalSimilarity`）
3. 目标一致度（`FinalGoalAlignment`）

评估流程分两阶段：
- 阶段A：从 groundtruth paper 提取 future idea（默认使用 `gpt-5.4`）
- 阶段B：评委团打分（`gpt-5.4 + claude-sonnet-4-6 + gemini-2.5-pro`）

---

## 1. 输入与目录约定

当前默认路径：
- Baseline 结果根目录：`/Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/results/baselines_output/Idea_forward`
  - 需包含：`<model>/<topic>/idea.json`
- Idea forward 评估输出目录：`/Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/results/benchmark_output/Idea_forward`
  - 需包含：`label_strategies.json`
- 源论文 PDF 目录：`/Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/data/papers_2025_v1`

---

## 2. 模型与网关配置

统一网关（OpenAI 兼容）：
- `base_url`: `http://34.13.73.248:3888/v1/chat/completions`
- `api_key`: 你的有效 key

脚本会自动把 `.../v1/chat/completions` 归一化到 `.../v1`。

脚本现在支持自动读取同目录 `.env`（默认路径：`benchmark/.env`），并把其中变量作为全局默认值。

- 模板文件：[.env.example](/Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/.env.example)
- 实际配置：[.env](/Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/.env)

也可以通过参数显式指定：

```bash
python forward_idea_benchmark.py --env-file /path/to/.env
```

建议使用环境变量避免在命令行明文出现 key：

```bash
export OPENAI_API_KEY='你的key'
```

---

## 3. 一次性全流程运行（推荐）

```bash
python /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/forward_idea_benchmark.py \
  --base-url http://34.13.73.248:3888/v1/chat/completions \
  --api-key "$OPENAI_API_KEY" \
  --extract-model gpt-5.4 \
  --judge-models "gpt-5.4;claude-sonnet-4-6;gemini-2.5-pro" \
  --top-k 3 \
  --output-dir /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/output
```

说明：
- `--top-k 3`：每个 baseline idea 会先检索 3 个候选论文，再选综合分最高的一条作为最终匹配。
- 提取与评估都默认启用。

---

## 4. 分阶段运行

### 4.1 只做阶段A（提取 future idea）

```bash
python /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/forward_idea_benchmark.py \
  --skip-eval \
  --extract-model gpt-5.4 \
  --base-url http://34.13.73.248:3888/v1/chat/completions \
  --api-key "$OPENAI_API_KEY" \
  --output-dir /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/output
```

### 4.2 复用缓存，只做阶段B（评委团打分）

```bash
python /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/forward_idea_benchmark.py \
  --skip-extraction \
  --judge-models "gpt-5.4;claude-sonnet-4-6;gemini-2.5-pro" \
  --base-url http://34.13.73.248:3888/v1/chat/completions \
  --api-key "$OPENAI_API_KEY" \
  --output-dir /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/output
```

### 4.3 强制重提取 paper future idea

```bash
python /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/forward_idea_benchmark.py \
  --refresh-paper-ideas \
  --extract-model gpt-5.4 \
  --base-url http://34.13.73.248:3888/v1/chat/completions \
  --api-key "$OPENAI_API_KEY" \
  --output-dir /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/output
```

---

## 5. 小样本调试（建议先跑）

```bash
python /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/forward_idea_benchmark.py \
  --baseline-filter gpt-5 \
  --limit 5 \
  --top-k 2 \
  --skip-extraction \
  --base-url http://34.13.73.248:3888/v1/chat/completions \
  --api-key "$OPENAI_API_KEY" \
  --output-dir /tmp/forward_benchmark_debug
```

参数含义：
- `--baseline-filter`: 只评某个 baseline（目录名包含该字符串）
- `--limit`: 只取前 N 个 `idea.json`

---

## 6. 输出文件说明

输出目录默认：`benchmark/output`

会生成：

1. `paper_future_ideas.jsonl`
- 每篇论文一条记录
- 包含：提取模型、原文片段、结构化提取结果、拼接后的 `ground_truth_idea`

2. `all_pair_scores.jsonl`
- 每个 baseline idea 与每个候选 paper 的完整评分
- 包含每个维度下每位评委的原始打分与理由

3. `selected_pair_scores.jsonl`
- 每个 baseline idea 的最佳候选（`overall_avg` 最大）

4. `summary.json`
- 按 baseline 模型聚合平均分
- 字段：
  - `PolicyAlignment_avg`
  - `CombinationalSimilarity_avg`
  - `FinalGoalAlignment_avg`
  - `overall_avg`

---

## 7. 聚合规则

### 7.1 单维度评委团聚合
对同一维度（如 `PolicyAlignment`）：
- 收集 3 位评委分数
- 取均值作为该维度最终分（`score`）
- 输出分歧（`score_std`）

### 7.2 单样本总体分
- `overall_avg = 3个维度分数均值`

### 7.3 模型总体分
- 对该 baseline 下所有样本的对应分数再取均值

---

## 8. 常见问题

1. `gemini-2.5-pro` 偶发空响应或非 JSON
- 脚本会重试；仍失败则该评委该维度记为 0，并保留 `raw_response`。
- 建议：保持 `temperature=0`，必要时增加 `--retries`。

2. 运行很慢
- 阶段A需要逐篇论文调用 `gpt-5.4`，首次会较慢。
- 建议先跑一次提取，后续用 `--skip-extraction` 复用缓存。

3. key 泄露风险
- 不要把 key 写入脚本或日志；优先用环境变量。

4. 基线文件未被发现
- 检查目录是否满足 `results/<baseline>/<topic>/idea.json`。

---

## 9. 参数速查

可用参数（`--help`）：
- 路径类：`--results-root`, `--output-dir`
- 网关类：`--base-url`, `--api-key`
- 提取阶段：`--extract-model`, `--extract-retries`, `--extract-max-pages`, `--extract-max-chars-per-page`, `--refresh-paper-ideas`
- 评估阶段：`--judge-models`, `--top-k`, `--temperature`, `--max-tokens`, `--retries`
- 采样类：`--limit`, `--baseline-filter`
- 开关类：`--skip-extraction`, `--skip-eval`



python /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/forward_idea_benchmark.py \
  --env-file /Users/ccw/Desktop/中科院项目/多智能体论文/benchmark_auto/benchmark/.env
  --limit 3
