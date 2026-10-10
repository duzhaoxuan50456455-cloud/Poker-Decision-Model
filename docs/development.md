# Development setup and proposed conventions

This repository is a skeleton. Module files contain placeholders only; the sole
test verifies package import. No data contract or architecture is implemented.

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

## Proposed boundaries

- `data.py`: record schema, loading, preprocessing, and group split utilities.
- `models/`: encoder and shared candidate scorer; no generative vocabulary head.
- `baselines.py`: simple baselines and eventual LLM comparison.
- `evaluation.py`: quality, calibration, and performance measurements.
- `apps/`: interactive presentation of package functionality.
- `configs/`: reviewed experiment settings and provenance.
- `notebooks/`: exploration; reusable logic moves into the package.

Keep data, weights, credentials, and generated reports out of Git. Pin backbone
revisions and save tokenizer settings when models are introduced. Use opponent
history available before each decision; group related hands/sessions in splits.

## Future testing strategy

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
