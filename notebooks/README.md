# Notebooks

Start with [Meeting 1: PokerBench pipeline](01_pokerbench_pipeline.ipynb), consolidated
from the team's eight unique Colab notebooks. It audits the preflop and postflop
training CSVs, constructs validated decision examples, makes a fixed-seed context
group split, and evaluates three exact-candidate CPU baselines. No GPU or API key.

From the repository root:

```sh
uv sync --locked --extra analysis
uv run --extra analysis jupyter lab notebooks/01_pokerbench_pipeline.ipynb
```

Select the project environment, restart the kernel, and run all cells. Defaults:
5,000 working examples per source, seed 42, 20% of distinct contexts for validation.
All source rows are validated before sampling. Source revisions and hashes are
pinned. Postflop contexts exclude future board cards. Any parse failure is saved
and stops preparation. The official test set is never loaded.

Reusable logic lives in `src/poker_decision/`, including a matching command-line
workflow: `uv run --extra analysis python -m poker_decision.pipeline`.
Generated data, split manifests, metrics, and five errors stay in ignored
`data/processed/pokerbench_meeting1/` and `artifacts/pokerbench_meeting1/`.

See [integration notes](../docs/meeting1-integration.md) for source contributions and
limitations. Keep notebook outputs cleared in Git and import the installed package
rather than modifying Python's import path.
