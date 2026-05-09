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

SupraMAS, short for **Supra Multi-Agent Scientist**, is the public reviewer-facing framework release for the manuscript:

> Mechanism-informed closed-loop on-demand design of high-temperature superconducting coated conductors

The framework converts literature-derived mechanisms, empirical rules, and process constraints into auditable research ideas, executable experimental protocols, and mechanism-informed optimization decisions for strongly constrained materials synthesis. This public repository exposes the **overall software architecture, module interfaces, benchmark artifacts, examples, and data-availability package** so reviewers can inspect the research workflow and reproduce reported data checks.

Detailed production agent prompts, retrieval heuristics, optimization internals, private service integrations, local credentials, and deployment code are intentionally excluded. The `src/supramas` package is therefore a compact framework skeleton rather than the full private implementation.

## What This Public Skeleton Shows

- **Strategy mining** module boundaries and strategy-tree output shape.
- **Idea innovation** orchestration interface and expected artifact flow.
- **Experiment construction** proposal/evaluator boundaries and protocol artifact shape.
- **Real-world verification** records through released case studies and data files.
- **Mechanism-informed Bayesian optimization** interface and reported optimization trajectory.
- **Interpretability tracing** input/output records and traceback artifact shape.

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

For local development or structural checks, the editable install is enough for the shipped tests and CLI smoke tests. The public package is an interface skeleton; full private end-to-end research execution is not included.

## Quick Start

Inspect the framework skeleton without making model calls:

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

Optionally write a framework-overview artifact:

```bash
supramas \
  --topic "Design REBCO coated conductors with enhanced flux pinning" \
  --config configs/supramas.example.yaml \
  --output-dir outputs \
  --write-overview
```

The command writes `outputs/framework_overview.json`. It does not run private production agents.

## Configuration

Start from the public example configuration:

```bash
cp configs/supramas.example.yaml configs/supramas.local.yaml
```

Then fill values locally if you are adapting the skeleton to your own implementation. Do not commit `configs/supramas.local.yaml`, `.env` files, API keys, model gateway URLs, Milvus credentials, or other secrets.

Important runtime services:

| Service | Purpose | Required for private/full run |
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
src/supramas/                    Public framework skeleton
  strategy_tree/                 Literature-grounded strategy-tree construction
  protocol_optimizer/            Protocol proposal, retrieval, and evaluator modules
  parameter_optimization/        Mechanism-informed Bayesian optimization utilities
  explainability/                Mechanism, protocol, idea, and strategy traceback
  prompts/                       Prompt-family catalog placeholders
  utils/                         Lightweight artifact helpers

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

This repository is a cleaned public framework/data release. It should not contain real API keys, private model gateways, local `.env` files, private Milvus credentials, private PDF corpora, full private implementation internals, or old Git history.

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
