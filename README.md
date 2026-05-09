<div align="center">
  <img src="assets/logo.png" alt="SupraMAS logo" width="720">

  <h1>SupraMAS</h1>

  <p>
    <strong>Mechanism-informed multi-agent research framework for literature-grounded,
    executable, and closed-loop design of high-temperature superconducting coated conductors.</strong>
  </p>

  <p>
    <a href="https://github.com/OpenMatSci/SupraMAS"><img alt="GitHub repository" src="https://img.shields.io/badge/GitHub-OpenMatSci%2FSupraMAS-181717?logo=github"></a>
    <a href="https://github.com/OpenMatSci/SupraMAS/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
    <img alt="Python" src="https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg">
    <a href="https://doi.org/10.57967/hf/8671"><img alt="Data DOI" src="https://img.shields.io/badge/Data-10.57967%2Fhf%2F8671-blue"></a>
  </p>
</div>

## Overview

SupraMAS, short for **Supra Multi-Agent Scientist**, is a public research-code release for the manuscript:

> Mechanism-informed closed-loop on-demand design of high-temperature superconducting coated conductors

The framework converts literature-derived mechanisms, empirical rules, and process constraints into auditable research ideas, executable experimental protocols, and mechanism-informed optimization decisions for strongly constrained materials synthesis. The released implementation focuses on high-temperature superconducting REBCO coated conductors and flux-pinning engineering, but the workflow is designed around reusable abstractions: strategy mining, expert-critic protocol refinement, interpretability tracing, and closed-loop parameter optimization.

This repository contains the core framework code, example inputs, benchmark artifacts, reproduction helpers, and the public data-availability package. Internal deployment code, private service integrations, local credentials, and private infrastructure are intentionally excluded from the public release.

## What SupraMAS Does

- **Strategy mining** builds a recursively expandable strategy tree from literature-grounded tuning dimensions, reported effects, limitations, and expected outcomes.
- **Idea innovation** assembles mechanism-aware strategy branches into candidate research hypotheses.
- **Experiment construction** converts candidate ideas into SOP-like experimental protocols through an expert-critic loop.
- **Real-world verification** records laboratory feedback and links measurements back to generated protocols.
- **Mechanism-informed Bayesian optimization** updates candidate synthesis parameters using mechanistic priors and historical experimental data.
- **Interpretability tracing** connects optimization decisions and protocol steps back to mechanism records, idea components, and strategy-tree evidence.

## Table Of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Repository Layout](#repository-layout)
- [Architecture](#architecture)
- [Benchmark](#benchmark)
- [Data Availability](#data-availability)
- [Verification](#verification)
- [Security And Release Hygiene](#security-and-release-hygiene)
- [Citation](#citation)
- [License](#license)

## Installation

SupraMAS requires **Python 3.10 or newer**.

```bash
git clone https://github.com/OpenMatSci/SupraMAS.git
cd SupraMAS

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For local development or structural checks, the editable install is enough for the shipped tests and CLI smoke tests. Full end-to-end research runs require local credentials and runtime services described in [Configuration](#configuration).

## Quick Start

Validate the CLI without making model calls:

```bash
supramas --help

supramas \
  --topic "Improve the bearing current capacity of superconducting strips through pinning" \
  --config configs/supramas.example.yaml \
  --dry-run
```

Run the bundled quickstart topic:

```bash
supramas \
  --topic "$(python -c 'import json; print(json.load(open("examples/quickstart/topic.json"))["topic"])')" \
  --config configs/supramas.example.yaml \
  --dry-run
```

After configuring an LLM endpoint, embedding endpoint, literature retrieval services, and vector database, remove `--dry-run` to execute the workflow:

```bash
supramas \
  --topic "Design REBCO coated conductors with enhanced flux pinning" \
  --config configs/supramas.local.yaml \
  --output-dir outputs \
  --max-tree-depth 3
```

Typical workflow outputs are written under `outputs/<topic>/` and include generated ideas, refined ideas, refined experimental protocols, and conversation logs.

## Configuration

Start from the public example configuration:

```bash
cp configs/supramas.example.yaml configs/supramas.local.yaml
```

Then fill values locally. Do not commit `configs/supramas.local.yaml`, `.env` files, API keys, model gateway URLs, Milvus credentials, or other secrets.

Important runtime services:

| Service | Purpose | Required for full run |
| --- | --- | --- |
| OpenAI-compatible LLM endpoint | Agent reasoning, self-revision, and protocol generation | Yes |
| Embedding endpoint | Literature and knowledge-library retrieval | Yes |
| Milvus | Vector search over papers, strategies, and templates | Yes |
| Semantic Scholar / Google Scholar compatible search | Literature discovery | Recommended |
| GROBID | PDF parsing for local literature workflows | Optional |
| Translation credentials | Cross-language literature processing | Optional |

Most values can be supplied through environment variables or YAML. See [`configs/supramas.example.yaml`](configs/supramas.example.yaml) and [`docs/configuration.md`](docs/configuration.md).

## Repository Layout

```text
src/supramas/                    Core Python package
  strategy_tree/                 Literature-grounded strategy-tree construction
  protocol_optimizer/            Protocol proposal, retrieval, and evaluator modules
  parameter_optimization/        Mechanism-informed Bayesian optimization utilities
  explainability/                Mechanism, protocol, idea, and strategy traceback
  prompts/                       Agent prompt templates
  utils/                         Formatting and conversion helpers

knowledge_libraries/             Mechanism, strategy, and template libraries
benchmark/                       Benchmark inputs, labels, evaluators, adapters, and results
data_availability/               Public manuscript data-availability package
examples/                        Quickstart topic and released case-study records
configs/                         Example runtime and benchmark configuration files
scripts/                         Reproduction, checksum, and manifest utilities
docs/                            Architecture, benchmark, configuration, data, and security notes
tests/                           Lightweight structural tests
```

## Architecture

SupraMAS is organized as a five-stage, human-in-the-loop research workflow:

1. **Strategy Mining**: recursively expands a strategy tree from literature-derived control dimensions, limitations, and expected outcomes.
2. **Idea Innovation**: combines selected strategy branches into mechanism-aware research hypotheses.
3. **Experiment Construction**: decomposes hypotheses into executable experimental protocols with expert-critic revision.
4. **Real-World Verification**: records experimental execution and measurements, including structural and superconducting-property feedback.
5. **Parameter Optimization**: integrates mechanistic priors and experimental data into a closed-loop optimization process.

The workflow is connected by shared memory and three external knowledge resources: a Knowledge Library, a Policy Library, and a Templates Library. For more detail, see [`docs/architecture.md`](docs/architecture.md).

## Benchmark

The public benchmark evaluates scientific-planning quality across two tracks:

- **Idea foresight**: strategy overlap and strategy-combination alignment against post-cutoff superconductivity studies.
- **Protocol evaluation**: experimental reproducibility and standalone feasibility of generated protocols.

Reported benchmark inputs, labels, and manuscript results are included under [`benchmark/`](benchmark/) and [`data_availability/benchmark_results/`](data_availability/benchmark_results/).

Run lightweight validation and summary commands:

```bash
python scripts/reproduce_benchmark.py --validate
python scripts/reproduce_benchmark.py --summary-only
python -m benchmark.evaluation.eval_idea --help
python -m benchmark.evaluation.eval_protocol --help
```

Full LLM-based re-evaluation requires local model credentials and may incur model-service costs. See [`benchmark/README.md`](benchmark/README.md) and [`docs/benchmark.md`](docs/benchmark.md).

## Data Availability

The manuscript data package is available in two forms:

- Local mirror in this repository: [`data_availability/`](data_availability/)
- Public data repository: <https://doi.org/10.57967/hf/8671>

The package includes source data for manuscript figures and tables, growth and annealing parameters, superconducting-property summaries, optimization trajectories, benchmark labels/results, expert-evaluation files, generated protocols, interpretability records, metadata, manifests, and checksums.

Verify the local data package:

```bash
python scripts/verify_checksums.py data_availability/metadata/checksums_sha256.csv
```

See [`data_availability/README.md`](data_availability/README.md) and [`docs/data_availability.md`](docs/data_availability.md).

## Verification

Recommended checks before opening a pull request or publishing a release:

```bash
python -m compileall src benchmark scripts
python scripts/reproduce_benchmark.py --validate
python scripts/verify_checksums.py data_availability/metadata/checksums_sha256.csv
```

If `pytest` is installed locally:

```bash
python -m pytest
```

## Security And Release Hygiene

This repository is a cleaned public release. It should not contain real API keys, private model gateways, local `.env` files, private Milvus credentials, private PDF corpora, or old Git history.

Before publishing a new release, scan for accidental secrets:

```bash
rg -n "sk-|AKLT|api[_-]?key|password|secret|token" .
```

Expected matches should be limited to placeholder keys, environment-variable readers, documentation, and non-sensitive benchmark text. See [`docs/security.md`](docs/security.md).

## Contributing

Contributions are welcome through issues and pull requests. Please keep changes focused, include a short description of the scientific or engineering motivation, and run the verification commands above before submitting.

For substantial changes, open an issue first so maintainers can discuss scope, data requirements, and reproducibility expectations.

## Citation

If you use SupraMAS, the released benchmark, or the associated data package, please cite the repository and the accompanying manuscript. Machine-readable citation metadata is provided in [`CITATION.cff`](CITATION.cff).

Public code release:

```text
https://github.com/OpenMatSci/SupraMAS/tree/main
```

Public data repository:

```text
https://doi.org/10.57967/hf/8671
```

## License

SupraMAS is released under the [MIT License](LICENSE).
