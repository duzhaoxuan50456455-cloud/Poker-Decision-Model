# Poker Decision Model

ACM AI Fall 2026. A context-conditioned poker decision model project, starting with
an audited PokerBench pipeline and CPU baselines over variable candidate actions.

## Run Meeting 1

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked --extra analysis
uv run --extra analysis jupyter lab notebooks/01_pokerbench_pipeline.ipynb
```

Select the project environment, restart the kernel, and run all cells. Or run the
same workflow from the repository root without Jupyter:

```sh
uv run --extra analysis python -m poker_decision.pipeline
```

This downloads only the pinned preflop and postflop **training** CSVs (about 90 MB),
checks their hashes, inspects and validates every row, and selects 5,000 examples
per source for a quick CPU run. It saves context-grouped train/validation subsets,
provenance, exact-candidate metrics, and five mistakes. The official test set is
never loaded. No GPU or API key is needed.

Use `--kind preflop` for the smaller source, `--skip-baseline` for data preparation,
or `--sample-per-kind 0` to use all validated rows. Outputs live in ignored
`data/processed/pokerbench_meeting1/` and `artifacts/pokerbench_meeting1/`.

Inputs are `DecisionInputs(context, candidates)`; labels are stored separately as
`target_index`. Exact action amounts are preserved. Postflop contexts include only
cards already dealt. See [integration notes](docs/meeting1-integration.md) for the
source notebook comparison, data contract, baseline design, and limitations.

## Development

Python 3.12 is the team default; 3.13 is supported. Core parsing and evaluation
need no optional ML dependencies. Install extras as needed: `analysis` (pandas,
sklearn, JupyterLab, plots), `ml` (torch, transformers), `demo` (Streamlit), and
`llm` (OpenAI SDK). Keep extra flags on `uv run` to retain those dependencies.

```sh
uv sync --locked
uv run --locked pytest
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
```

The learned-baseline tests run when the analysis extra is installed; CI checks
both the core environment and the learned baseline. Raw data and generated
reports stay out of Git.

## Layout

```text
src/poker_decision/   data contract, preprocessing, baselines, evaluation, workflow
notebooks/           consolidated Meeting 1 walkthrough
apps/                demo placeholder
tests/               offline parsing, leakage, splitting, metrics, model tests
data/raw/            local source training CSVs (ignored)
data/processed/      generated datasets, failure reports, manifest (ignored)
artifacts/           generated metrics and mistakes (ignored)
configs/             future experiment settings
docs/                project context, development notes, integration decisions
```

See [project context](docs/project-context.md), [development guide](docs/development.md),
and [roadmap](docs/roadmap.md). Neural model training, opponent adaptation, and the
demo remain future work. Baseline scores measure reference-action agreement.
