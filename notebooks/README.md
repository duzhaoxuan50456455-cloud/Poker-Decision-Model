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

## Meeting 2

Continue with [Meeting 2: single-pass decision model](02_pokerbench_decision_model.ipynb),
consolidated from three shared Colabs and the local `week2.ipynb` submission.

```sh
uv sync --locked --extra analysis --extra ml
uv run --locked --extra analysis --extra ml jupyter lab notebooks/02_pokerbench_decision_model.ipynb
```

The notebook explains the input adapter, joint serialization, candidate token spans,
mean pooling, shared head, candidate padding, and raw-logit cross-entropy. It requires
a tiny training-set learning check, resets the model for real training, selects a
checkpoint by validation log loss, reloads it, compares all baselines, shows five
mistakes, and measures candidate-order sensitivity. Encoder revisions are pinned.

Use Colab's GPU runtime or a local ML environment. The Colab bootstrap clones the
repository only if needed; the Meeting 2 changes must be published before that
route works. Existing local checkouts work immediately. `POKER_SMOKE=1` before
launching Jupyter selects 128 train / 64 validation rows for quick verification;
all baselines are refitted on that selected training subset. Normal runs reuse
the complete saved Meeting 1 membership. If absent, preparation uses Meeting 1 defaults.

Generated checkpoints, split copies, training history, comparisons, probabilities,
and errors live under `artifacts/pokerbench_meeting2/<backbone>-<timestamp>/`.
No official test data is loaded and no tokenization failures are silently skipped.
See [Meeting 2 integration notes](../docs/meeting2-integration.md).
