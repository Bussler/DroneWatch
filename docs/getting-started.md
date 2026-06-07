# Getting Started

## Prerequisites

DroneWatch currently assumes:

- Python 3.11 or newer
- `uv`
- Rust and Cargo
- Docker only if you want the validation image or MLflow service through Compose

## Install the Project

The main local setup command is:

```bash
make install
```

This performs two steps:

```bash
uv sync --dev
uv run maturin develop -m rust/swarm_sim/Cargo.toml
```

## Fast Smoke Path

Use this path to confirm that Python, Rust, RLlib, and the core CLIs all work before starting a longer training run.

```bash
uv run python -c "import dronewatch; print(dronewatch.__version__)"
uv run python -c "from dronewatch.sim import rust_version; print(rust_version())"
make rollout-rust
make rollout-random
make render-random
make ppo-smoke
make tune-ppo-smoke
```

Expected outputs:

- `make rollout-random` writes `artifacts/reports/random_policy_report.json`
- `make render-random` also writes `artifacts/gifs/random_policy_episode.gif`
- `make ppo-smoke` writes `artifacts/reports/ppo_smoke_report.json`
- `make tune-ppo-smoke` writes a Tune search summary under `artifacts/reports/`

## Main Workflows

### Train PPO

```bash
make train-ppo
```

### Evaluate a checkpoint

```bash
make evaluate-ppo CHECKPOINT=artifacts/checkpoints/ppo/path-to-checkpoint
```

### Run Ray Tune search

```bash
make tune-ppo
```

### View MLflow runs

```bash
make mlflow-up
```

Or run the UI directly:

```bash
make mlflow-ui
```

### Run tests

```bash
make test
```

## Recommended Reviewer Path

If you are reading the repository to understand the project rather than to modify it, this sequence is usually enough:

1. Install with `make install`.
2. Run `make rollout-random`.
3. Run `make ppo-smoke`.
4. Open the report artifacts in `artifacts/reports/`.
5. Read the Results guide to interpret the output.
