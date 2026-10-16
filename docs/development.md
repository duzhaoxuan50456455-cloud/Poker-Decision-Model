# Development setup and proposed conventions

Meeting 1 implements the PokerBench data contract, preprocessing, grouped splits,
CPU candidate baselines, and evaluation. The neural model and demo remain
placeholders. See [integration notes](meeting1-integration.md) for the current workflow.

## Dependencies

Use a `src/` Python package and Python 3.12 as the team default. `pyproject.toml`
declares dependency ranges; `uv.lock` records exact versions. CI uses `--locked`
to detect stale lockfiles. NumPy and Pydantic are the proposed core dependencies.
Optional extras separate model work (torch, transformers), analysis (pandas,
sklearn, notebooks, plots), demo (Streamlit), and an LLM client (OpenAI).

Use `uv add PACKAGE`, `uv add --optional ml PACKAGE`, or `uv add --dev PACKAGE`.
Commit the manifest and lockfile together. Upgrade deliberately with
`uv lock --upgrade-package PACKAGE`, then run checks. Avoid a second requirements
file. Choose the poker variant and data source before adding simulator/solver
packages. Document accelerator-specific PyTorch setup when training begins.

Reference: [uv dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)
and [locking](https://docs.astral.sh/uv/concepts/projects/sync/).

## Package boundaries

- `data.py`: answer-free input schema, loading, preprocessing, validation, and grouped splits.
- `pipeline.py`: reproducible preparation, inspection, output manifests, and CLI.
- `models/`: encoder and shared candidate scorer; no generative vocabulary head.
- `baselines.py`: simple baselines and eventual LLM comparison.
- `evaluation.py`: quality, calibration, and performance measurements.
- `apps/`: interactive presentation of package functionality.
- `configs/`: reviewed experiment settings and provenance.
- `notebooks/`: exploration; reusable logic moves into the package.

## Notebooks in GitHub

Keep one maintained walkthrough notebook for Meeting 1. It is teaching material
that shows raw rows, decision examples, validation, metrics, and mistakes. Put
reusable preprocessing, splitting, training, and evaluation in the Python package
with tests; the notebook and command-line runner should call that same code.
Future training scripts and the demo should import the package directly.

Before committing a notebook, restart its kernel and run all cells, then clear
outputs and execution counts. Commit the notebook source and supporting package
changes together. Keep downloaded data, generated datasets, split manifests,
metrics, and model files in the ignored output directories. Preserve links to the
team's original notebooks in the integration notes instead of keeping competing
copies of the pipeline in the repository.

Keep data, weights, credentials, and generated reports out of Git. Pin backbone
revisions and save tokenizer settings when models are introduced. Use opponent
history available before each decision; group related hands/sessions in splits.

## Testing strategy

Pytest and coverage are configured; Ruff handles formatting/lint and mypy handles
strict package typing. CI runs on Python 3.12 and 3.13. Add meaningful tests as
behavior is implemented: schema failures, split leakage, variable option counts,
permutation equivariance, legal-option masks, gradient flow, known metric values,
tiny training loss reduction, and checkpoint round trips. Keep routine tests
CPU-only and offline; separate pretrained downloads and live API calls.

Future evaluation should include agreement, log loss, Brier score, calibration,
latency, throughput, and opponent-history ablations. Fit calibration on validation
only and preserve an untouched test set. Record hardware and workload when
benchmarking; compare identical candidate options across systems.
