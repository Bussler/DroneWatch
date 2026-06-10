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

## Build and Run with Docker

The repository includes a Docker image that starts PPO training directly.

Build it from the repository root:

```bash
docker build -t dronewatch:train-entrypoint .
```

The image copies the repository into `/app` and uses this entrypoint:

```bash
uv run python -m dronewatch.training.train_ppo
```

That means arguments added after the image name in `docker run` are passed straight to the training script.

### Run the default training config

```bash
docker run --rm dronewatch:train-entrypoint
```

### Choose a different experiment file

```bash
docker run --rm dronewatch:train-entrypoint --config configs/debug.yaml
```

### Pass additional config overrides

```bash
docker run --rm dronewatch:train-entrypoint \
	--config configs/config.yaml \
	training.stop.iterations=10 \
	model=ppo_feedforward
```

### Persist checkpoints, reports, and MLflow outputs on the host

By default, container-local outputs disappear when the container is removed. Mount the output directories if you want to keep them:

```bash
docker run --rm \
	-v "$(pwd)/artifacts:/app/artifacts" \
	-v "$(pwd)/outputs:/app/outputs" \
	dronewatch:train-entrypoint \
	--config configs/debug.yaml
```

This keeps checkpoint, report, GIF, and MLflow output files in the local repository directories.
