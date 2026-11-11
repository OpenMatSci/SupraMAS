# SupraMAS Data Availability

Current manuscript results are synchronized from the 1027 supporting-data package. The complete 130-file package and ZIP are available on [Hugging Face](https://huggingface.co/datasets/cuichaowei/supramas-data-availability/tree/main).

- `benchmark_results/`: final two-judge scores, inputs, labels, statistical estimates and comparisons, bootstrap arrays, prompts and rank-concordance analyses.
- `expert_evaluation/`: five original workbooks and individual and aggregated scores.
- `optimization_trajectory/`: seven process conditions and reported Tc/Jc summaries. Tc is the mean M–T value from three independent films; individual Tc values and standard deviations are not supplied in these summaries.
- `generated_protocols/` and `interpretability/`: generated protocols, feedback, and the supplied Chain 3 traceback report.
- `knowledge_libraries/`: released knowledge entries.
- `source_data/figure_source_data/`: original measurement exports with earlier figure labels, retained without changing raw measurements. `source_data/main_tables/` contains the current reported summary values.
- `metadata/`: inventory and checksums for this local GitHub selection.

The two large judge archives (`grouped_judge_outputs.jsonl` and `per_idea_judge_aggregates.json`) are hosted on Hugging Face. `idea_foresight/*/evaluation.jsonl` contains a compact selection of final judge scores, matched-label IDs and combination-match metadata. Protocol `evaluation_*.json` files contain the final case/judge records. These replace the earlier single-run result schema. Statistical source tables retain the case-level scores needed to verify the reported means.

```bash
python scripts/verify_checksums.py data_availability/metadata/checksums_sha256.csv
```
