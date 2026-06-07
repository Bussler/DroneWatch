# Getting Started

## Prerequisites

DroneWatch currently assumes:

- Python 3.11 or newer
- `uv` for package management
- Rust and Cargo
- Docker only if you want the validation image or MLflow service through Compose

## Install the Project

The main local setup command is:

```bash
make install
```

This performs the python and rust project installation:

```bash
uv sync --dev
uv run maturin develop -m rust/swarm_sim/Cargo.toml
```

## Main Workflows
Most commands can be run efficiently via a makefile.  
See all available commands with `make help`

### Train PPO

```bash
make train-ppo
```

The training prints the results of each training iteration to the command line and logs out to mlflow.  
Periodic evaluations are run during training.  
Artifacts of these evaluations (checkpoints, gifs, reports) are stored in `/artifacts/`.  
If not configured otherwise, mlflow logs are stored in `/outputs/mlruns/`.

### Evaluate a checkpoint

```bash
make evaluate-ppo CHECKPOINT=artifacts/checkpoints/ppo/path-to-checkpoint
```

Similar to the training script, logging and evaluations outputs are stored in `/outputs/mlruns/` and `/artifacts`.

### Run Ray Tune search

```bash
make tune-ppo
```

Similar to the training script, logging and evaluations outputs are stored in `/outputs/mlruns/` and `/artifacts`.

### View MLflow runs

```bash
make mlflow-up
```

### Run tests

```bash
make test
```

### Run a baseline random policy

```bash
make rollout-random
```
