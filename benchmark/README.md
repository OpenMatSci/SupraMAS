# SupraMAS Benchmark

This directory contains the benchmark artifacts used for prospective scientific-planning evaluation.

## Contents

- `data/`: benchmark topics, post-cutoff paper ideas, idea-foresight labels, and protocol labels.
- `evaluation/`: LLM-based evaluation scripts.
- `results/`: reported benchmark outputs from the manuscript.
- `baselines/`: lightweight baseline adapters.
- `prompts/`: judge and extraction prompt templates.

## Quick Checks

```bash
python benchmark/reproduce_benchmark.py --validate
python -m benchmark.evaluation.eval_idea --help
python -m benchmark.evaluation.eval_protocol --help
```

Running a full re-evaluation requires local model credentials configured outside Git.


## Current result records

Final results average Claude and GPT within each case, then average 32 idea topics or 31 protocol cases. `results/idea_foresight/*/evaluation.jsonl` stores one compact record per topic and judge; protocol JSON files store one record per case and judge. `benchmark_summary.csv` reports the four metrics separately. Full rationale archives and statistical analyses are linked in [the data index](../data_availability/README.md).
