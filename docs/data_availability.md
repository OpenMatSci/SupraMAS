# Data Availability

See [the local data index](../data_availability/README.md) for current results and the retained original measurement exports. The full supporting-data package and ZIP are on [Hugging Face](https://huggingface.co/datasets/cuichaowei/supramas-data-availability/tree/main). The GitHub selection omits the two large idea-judge rationale archives.

```bash
python scripts/verify_checksums.py data_availability/metadata/checksums_sha256.csv
```

## Current manuscript figures

[View all 4 main figures and 13 supplementary figures](../assets/figures/README.md), named according to the current manuscript and SI. Original measurement exports in the data directory are separate from these publication figures.

## GitHub and Hugging Face paths

The repositories share current reported results but are not identical file mirrors. Hugging Face hosts the complete 130-file supporting-data package; GitHub includes selected results and additional original measurement exports.

| Content | GitHub path | Hugging Face path |
| --- | --- | --- |
| Statistical source tables | `data_availability/benchmark_results/statistical_analysis/` | `SupraMAS_Supporting_Data/benchmark_results/statistical_analysis/` |
| Ranking and sensitivity analyses | `data_availability/benchmark_results/expert_concordance/` | `SupraMAS_Supporting_Data/benchmark_results/expert_concordance/` |
| Expert ratings | `data_availability/expert_evaluation/` | `SupraMAS_Supporting_Data/expert_evaluation/` |
| Process and performance summaries | `data_availability/optimization_trajectory/` | `SupraMAS_Supporting_Data/optimization_trajectory/` |
| Protocols, traceback and knowledge entries | `data_availability/{generated_protocols,interpretability,knowledge_libraries}/` | `SupraMAS_Supporting_Data/{generated_protocols,interpretability,knowledge_libraries}/` |
| Complete idea-judge output archives | Linked from `data_availability/README.md`; compact records in `benchmark/results/idea_foresight/` | `SupraMAS_Supporting_Data/benchmark_results/idea_evaluation/judge_outputs/` |
| Original experiment curve/image exports | `data_availability/source_data/figure_source_data/` (original measurement filenames) | Not included in the current package |
| Inventory and checksums | `data_availability/metadata/file_manifest.csv` and `checksums_sha256.csv` | `SupraMAS_Supporting_Data/manifest.json` and `checksums_sha256.txt` |

The inventory files cover each repository's own selection. CSV line endings may differ while parsed values remain the same. The two large idea-judge archives are required by the statistical recomputation scripts; run those scripts from the complete downloaded package.

[Browse the complete package](https://huggingface.co/datasets/cuichaowei/supramas-data-availability/tree/main/SupraMAS_Supporting_Data) · [Download ZIP](https://huggingface.co/datasets/cuichaowei/supramas-data-availability/resolve/main/SupraMAS_Supporting_Data.zip?download=true)
