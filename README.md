# SupraMAS

SupraMAS is a mechanism-informed multi-agent research framework for literature-grounded idea generation, experimental protocol construction, interpretability tracing, and closed-loop optimization of high-temperature superconducting coated conductors.

This repository is the cleaned public release for the manuscript:

> Mechanism-informed closed-loop on-demand design of high-temperature superconducting coated conductors

## Repository Layout

```text
src/supramas/          Core SupraMAS package
knowledge_libraries/   Runtime mechanism, strategy, and template libraries
benchmark/             Benchmark inputs, evaluation scripts, adapters, and reported results
data_availability/     Manuscript data availability package
examples/              Quickstart topic and released case-study records
scripts/               Reproduction, checksum, and manifest utilities
docs/                  Architecture, configuration, benchmark, data, and security notes
tests/                 Lightweight structural tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For a minimal structural check without installing every optional research dependency:

```bash
PYTHONPATH=src python -m supramas.cli --help
python -m compileall src benchmark scripts
```

## Configuration

Copy the example configuration and fill in local credentials:

```bash
cp configs/supramas.example.yaml configs/supramas.local.yaml
```

The public repository intentionally does not include real API keys, private model gateways, Milvus credentials, or local `.env` files. Runtime credentials should be supplied through environment variables or a local config file that is ignored by Git.

## Quick Start

```bash
PYTHONPATH=src python -m supramas.cli \
  --topic "Improve the bearing current capacity of superconducting strips through pinning" \
  --config configs/supramas.example.yaml \
  --dry-run
```

Remove `--dry-run` only after configuring the LLM gateway, translation credentials if needed, vector database, and literature retrieval services.

## Benchmark

Benchmark inputs and labels are available in `benchmark/data/`, while the manuscript-reported results are in `benchmark/results/`.

```bash
python scripts/reproduce_benchmark.py --summary-only
python scripts/verify_checksums.py data_availability/metadata/checksums_sha256.csv
```

The full third-party `gpt-researcher` project is not vendored. See `benchmark/baselines/gpt_researcher_adapter/README.md` for the adapter policy.

## Data Availability

The `data_availability/` directory contains source data, optimization trajectories, benchmark labels/results, expert evaluation files, generated protocols, interpretability records, and dataset metadata prepared for the manuscript.

## Citation

If you use SupraMAS or the released data, please cite this repository and the associated manuscript. A machine-readable citation file is provided in `CITATION.cff`.

## License

This project is released under the MIT License.

