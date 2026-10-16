# Tests

Run `uv run --locked pytest` for offline parsing, leakage, split, probability,
metric, and preparation tests. Install the analysis extra and use
`uv run --locked --extra analysis pytest` to include the learned scorer tests.
CI runs both configurations. No test downloads source data or calls an API.

The real-data workflow is also verified by running the consolidated notebook
from a fresh kernel. It checks both full training CSVs before sampling and saves
its provenance and validation results under ignored output directories.
