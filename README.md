# Poker Decision Model

ACM AI Fall 2026. Project skeleton for a context-conditioned poker decision model.
No model, data pipeline, evaluation, baseline, or demo is implemented yet.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Python 3.12 is the default; 3.13 is supported. Optional extras are reserved for
future work: `ml` (torch, transformers), `analysis` (pandas, sklearn, JupyterLab,
matplotlib), `demo` (Streamlit), and `llm` (OpenAI SDK). Install only what you need:

```sh
uv sync --locked --extra ml
```

Use multiple `--extra` flags as needed. Keep those flags on `uv run` to retain
extras. The default setup needs no model downloads or API credentials.

## Layout

```text
src/poker_decision/   empty module placeholders
tests/              package import smoke test and future testing notes
data/examples/      reserved for small fixtures
data/raw/           local source data (ignored)
data/processed/     generated datasets (ignored)
configs/            future experiment settings
notebooks/          future exploration
apps/               demo placeholder
docs/               supplied context, setup decisions, roadmap
artifacts/          generated outputs (ignored)
```

See [project context](docs/project-context.md), [development guide](docs/development.md),
and [roadmap](docs/roadmap.md). CI checks lint, formatting, typing, and package import
on Python 3.12 and 3.13.
