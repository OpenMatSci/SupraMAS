---
pretty_name: SupraMAS Data Availability Package
language:
- en
tags:
- materials-science
- superconductivity
- high-temperature-superconductors
- multi-agent-systems
- bayesian-optimization
- benchmark
- tabular
---

# SupraMAS Data Availability Package

This package contains the data files prepared for the manuscript:

Mechanism-informed closed-loop on-demand design of high-temperature superconducting coated conductors

## Package Contents

### source_data/

Machine-readable source data for the main manuscript tables, experimental figures and experimental summaries.

- `source_data/main_tables/growth_parameters.csv`: growth and annealing parameters reported in the main manuscript.
- `source_data/main_tables/superconducting_properties.csv`: superconducting properties reported in the main manuscript.
- `source_data/figure_source_data/`: numerical source data and image source files underlying the experimental figure panels:
  - `Fig3_single_pinning_system/`: R-T curves, Jc-B curves, M-H hysteresis data, Jc calculation details and SEM thickness image for the single artificial pinning-system films.
  - `Fig4_multicomponent_films/`: M-T curves, R-T curves, Jc-B curves and M-H hysteresis data for the initial and optimised multicomponent films.
  - `Supplementary_Fig1_struct_morphology/`: XRD scans and AFM morphology images for the structural and morphological characterisation of the optimised multicomponent films.

### optimization_trajectory/

- `optimization_trajectory.csv`: closed-loop optimisation trajectory, including process parameters, performance metrics and mechanism-level interpretation.
- `mibo_report.pdf`: detailed mechanism-informed Bayesian optimisation report.

### benchmark_data/

Benchmark inputs and labels used for prospective scientific-planning evaluation.

- `inputs/benchmark_topics.json`: 32 benchmark topic prompts.
- `inputs/post_cutoff_paper_ideas.json`: structured post-cutoff paper-derived ideas.
- `labels/idea_foresight_label_strategies.json`: structured label strategies used for idea-foresight evaluation.
- `labels/protocol_reference_labels.json`: reference experimental workflows used for protocol-level evaluation.

### benchmark_results/

Automatic evaluation outputs.

- `benchmark_summary.csv`: compact summary of idea-foresight and protocol-level benchmark results for SupraMAS and the five baseline systems reported in the manuscript.
- `idea_foresight/`: per-model idea-foresight evaluation results and extracted predicted strategies.
- `protocol_evaluation/`: per-model experimental feasibility and reproducibility evaluation results.

### expert_evaluation/

Human expert assessment data.

- `result_1.xlsx` to `result_5.xlsx`: anonymized expert scoring tables for SupraMAS and the five baseline systems reported in the manuscript.
- `human_expert_mean_scores.csv`: mean expert scores by model and dimension for the same systems.
- `human_eval_mean_scores.json`: JSON version of the aggregated expert scores.
- `human_template.md`: expert evaluation template.

### knowledge_libraries/

Structured knowledge sources used by SupraMAS.

- `mechanism/mechanism.json`: process-parameter and mechanism-response knowledge entries.
- `strategy/tuning_strategy_v1.json`: tuning-strategy database.
- `template/*.json`: experimental and mechanism templates.

### generated_protocols/

AI-generated ideas, strategy trees and experimental protocols used in the laboratory-execution case studies.

- `single_apc_systems/`: EuBCO+BYNO, EuBCO+BYTO/Cu and YBCO+Y211 protocol records.
- `multivariable_system/`: multi-rare-earth REBCO + BZO + BYNO protocol and human-feedback records.

### interpretability/

Mechanism-aware backtracking records for the interpretability analysis.

- `input/final_protocol.json`: experimental protocol input used by the explainability-tracking module.
- `input/mechanism_analysis.md`: mechanism-analysis input used by the explainability-tracking module.
- `input/tree_formatted.json`: strategy-tree input used by the explainability-tracking module.
- `output/traceback_report.md`: generated traceback report used for the interpretability case study.

### metadata/

Dataset-level metadata and indices.

- `file_manifest.csv`: file list and sizes for deposited data files.
- `checksums_sha256.csv`: SHA256 checksums for deposited files, excluding the checksum table itself.
- `knowledge_library_counts.csv`: record counts for each structured knowledge file.
- `post_cutoff_reference_papers.csv`: metadata and DOI list for the post-cutoff papers used to construct benchmark labels.

## Source Data Notes

Experimental figure source data are provided as `.xlsx` numerical tables for transport, magnetisation, critical-current-density and XRD data, with SEM/AFM image source files supplied where the figure panels are image-based. For the `.xlsx` files, `Sheet1` contains the populated source data; additional blank sheets, where present, are retained from the original export format.
