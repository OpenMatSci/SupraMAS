# GPT Researcher Adapter

The full GPT Researcher project is intentionally not copied into SupraMAS-public. This keeps the repository small and avoids mixing third-party source history and license maintenance into the core SupraMAS release.

To reproduce GPT Researcher baselines:

1. Install GPT Researcher from its upstream repository according to its license and documentation.
2. Generate protocols for the topics in `benchmark/data/benchmark_topics.json`.
3. Export results using the same JSON layout as `benchmark/results/idea_foresight/gpt_researcher/` and `benchmark/results/protocol_evaluation/gpt_researcher/`.
4. Run the evaluation scripts with your local benchmark configuration.

