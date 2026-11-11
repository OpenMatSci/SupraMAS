# SupraMAS Benchmark 使用说明

本说明对应当前投稿版的固定评分数据。完整数据包在 [Hugging Face](https://huggingface.co/datasets/cuichaowei/supramas-data-availability/tree/main/SupraMAS_Supporting_Data)，GitHub 的结果入口为 [`results/`](results/)。

## 评价范围

六个系统为 SupraMAS、GPT-Researcher、GPT-5、Gemini-2.5-Pro-Thinking、Grok-4 和 DeepSeek-V3。

| 评价轨道 | 案例数 | 指标 |
| --- | ---: | --- |
| Idea 评估 | 32 个主题 | strategy_overlap、strategy_combination |
| 协议评估 | 31 个论文来源案例 | experimental_reproducibility、experimental_feasibility |

每个指标均由 Claude 和 GPT 两位固定评委评分。先在每个主题或案例内对两位评委等权平均，再在主题或案例间平均。共保留 1,512 个指标评分。论文分别报告四项指标。

策略重合度为匹配的唯一参考策略数除以全局参考库的 155 条策略，评委分数保留四位小数。策略组合分为每个预测 idea 对全部 31 个参考组的组合相似度最高分。两种协议指标在同一系统、同一案例内使用相同的冻结协议文本；只有可复现性评委获得参考协议。

五个 baseline 的可行性评分来自另一次网关运行，SupraMAS 沿用原评分；相关对比描述现存记录，不能单独分离系统与评分运行的影响。

## 结果文件

- [`results/benchmark_summary.csv`](results/benchmark_summary.csv)：六系统、四项指标的均值。
- `results/idea_foresight/<system>/evaluation.jsonl`：每个主题、每位评委的紧凑记录，包括最终分数和匹配标签编号。
- `results/protocol_evaluation/<system>/evaluation_reproducibility.json` 和 `evaluation_feasibility.json`：每个案例、每位评委的最终协议评分记录。
- `evaluation_summary.json`：对应系统的汇总均值。
- [`../data_availability/benchmark_results/statistical_analysis/`](../data_availability/benchmark_results/statistical_analysis/)：逐例评分、均值和区间、配对比较及 bootstrap 数组。
- [`../data_availability/benchmark_results/evaluation_protocol/`](../data_availability/benchmark_results/evaluation_protocol/)：提示词、rubric 和模型版本说明。

完整分组输出及完整 idea 评判理由保存在 Hugging Face 的 `benchmark_results/idea_evaluation/judge_outputs/`。GitHub 的紧凑记录用于读取最终分数，不替代完整理由归档。

## 验证已发布结果

在仓库根目录运行，以下检查不调用模型：

```bash
python scripts/reproduce_benchmark.py --validate
python scripts/reproduce_benchmark.py --summary-only
python scripts/verify_checksums.py data_availability/metadata/checksums_sha256.csv
```

查看公开评价程序的参数：

```bash
python -m benchmark.evaluation.eval_idea --help
python -m benchmark.evaluation.eval_protocol --help
```

重新调用评价模型需要自行配置凭据；新运行不会自动重建已冻结的评分记录。

## 从冻结评分重算统计

下载完整 ZIP，并在解压后的数据包内运行统计脚本，使脚本能够找到完整评分输入：

```bash
bash scripts/download_data.sh
unzip SupraMAS_Supporting_Data.zip
cd SupraMAS_Supporting_Data
python verify_package.py
python -m pip install -r benchmark_results/requirements.txt
python benchmark_results/statistical_analysis/recompute_statistics.py
python benchmark_results/expert_concordance/recompute_concordance.py
```

自动评估采用 20,000 次案例级配对 bootstrap、逐项 95% BCa 区间，以及 100,000 次双侧配对符号翻转检验。两评委均值比较使用 20 项 Holm 校正；逐评委分析使用独立的 40 项校正。详细方法与固定随机种子见数据包。

GitHub 与 Hugging Face 的目录对应关系见[数据索引](../docs/data_availability.md#github-and-hugging-face-paths)。
