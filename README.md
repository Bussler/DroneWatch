# DroneWatch
[Documentation](https://bussler.github.io/DroneWatch/) |


DroneWatch is a multi-agent reinforcement learning project for cooperative 2D search. A shared-policy PPO controller learns to coordinate drones that must discover targets in a partially observable continuous world with local sensing, short-range communication, collisions, and circular no-fly obstacles.

The project supports scripted rollouts, a random-policy baseline, shared-policy PPO training, checkpoint evaluation, and local Ray Tune hyperparameter search.
Task progress is tracked with task metrics, not reward alone: target discovery rate, discovered target count, coverage ratio, collision count, obstacle violations, connectivity ratio, average communication neighbors, episode length, and success rate.

The current implementation combines:

- a Rust simulator for world state, geometry, collisions, coverage, and connectivity metrics
- a Python RLlib training and evaluation stack built around `SwarmSearchEnv`
- structured OmegaConf configuration with Pydantic validation
- local MLflow logging, JSON reports, GIF rendering, and Ray Tune sweeps

![Obstacle avoidance rollout](docs/assets/gifs/obstacle_avoidance_2.gif)

## Key Findings

- Feedforward PPO was useful for validating the training stack, but it did not solve the full search problem. LSTM memory was necessary for robust search behavior.
- Reward rebalance and PPO stabilization mattered as much as model choice. Lower entropy, tighter action-noise clipping, smaller PPO updates, and reward rescaling materially improved training.
- Mixed rewards worked best: global terms promoted mission success and coverage, while local terms gave each drone clearer credit for discoveries, collisions, and obstacle violations.

The trained agent solves the final obstacle-heavy task and generalizes well to new scenarios with randomized positions of targets, drones and obstacles.
This makes DroneWatch a useful reference for how recurrent policies, reward shaping, and PPO stabilization interact in partially observable cooperative search.

For a more thorough ablation study and summarization, look at [Documentation Results](https://bussler.github.io/DroneWatch/results/).

## Quick Start

### Prerequisites

- Python 3.11 or newer
- `uv`
- Rust toolchain with Cargo
- Docker, optional for the MLflow UI and validation image

### Install

```bash
make install
```

This installs Python dependencies and builds the Rust extension.

### Run a small PPO smoke path

```bash
uv run python -m dronewatch.training.train_ppo --config configs/debug.yaml
```

### Run the main workflows

```bash
make train-ppo
make evaluate-ppo CHECKPOINT=artifacts/checkpoints/ppo/path-to-checkpoint
make tune-ppo
make test
```

## Main Workflows

### Training

Run local PPO training with the default root config:

```bash
make train-ppo
```

### Checkpoint evaluation

```bash
make evaluate-ppo CHECKPOINT=artifacts/checkpoints/ppo/path-to-checkpoint
```

### Ray Tune sweep

```bash
make tune-ppo
```

Tune uses the dedicated root config at `configs/tune_ppo.yaml` and optimizes the metric configured at `tune.metric`, currently `target_discovery_rate`.

### MLflow UI

```bash
make mlflow-up
```

The local tracking store is `outputs/mlruns/`.

## Configuration

DroneWatch uses OmegaConf to compose YAML configs and Pydantic to validate the resolved result. The main root configs are:

- `configs/config.yaml` for PPO training
- `configs/debug.yaml` for a minimal smoke path
- `configs/evaluate.yaml` for standalone checkpoint evaluation
- `configs/random_policy.yaml` for the random-policy baseline
- `configs/tune_ppo.yaml` for Ray Tune sweeps

Group configs live under `configs/env`, `configs/model`, `configs/training`, `configs/evaluation`, `configs/random_policy`, `configs/rendering`, `configs/logging`, and `configs/tune`.

CLI overrides can replace config groups or individual fields. Example:

```bash
uv run python -m dronewatch.training.train_ppo \
	--config configs/config.yaml \
	model=ppo_feedforward \
	training.stop.iterations=1 \
	training.ppo.minibatch_size=64
```

Each run writes a resolved config snapshot named `resolved_config.yaml` beside its artifacts.

## Artifacts

By default, DroneWatch writes outputs to:

- `artifacts/reports/` for JSON reports
- `artifacts/gifs/` for rendered episodes
- `artifacts/checkpoints/ppo/` for PPO checkpoints
- `outputs/mlruns/` for MLflow runs

## Build and run with Docker

Build the training image:

```bash
docker build -t dronewatch:train-entrypoint .
```

The container entrypoint is the PPO trainer, so arguments passed to `docker run` are forwarded directly to `python -m dronewatch.training.train_ppo`.

Run training with the default config inside the image:

```bash
docker run --rm dronewatch:train-entrypoint
```

Select a different experiment config:

```bash
docker run --rm dronewatch:train-entrypoint --config configs/debug.yaml
```

Pass extra OmegaConf overrides:

```bash
docker run --rm dronewatch:train-entrypoint \
	--config configs/config.yaml \
	training.stop.iterations=10 \
	project.seed=7
```

If you want training artifacts on the host, mount the artifact directories:

```bash
docker run --rm \
	-v "$(pwd)/artifacts:/app/artifacts" \
	-v "$(pwd)/outputs:/app/outputs" \
	dronewatch:train-entrypoint \
	--config configs/debug.yaml
```

## Documentation

The [MkDocs](https://bussler.github.io/DroneWatch/) site covers setup, configuration, the training pipeline, Rust environment details, troubleshooting, and result interpretation.

Build it locally with:

```bash
make docs-serve
```

For more detail, start with:

- [getting-started](https://bussler.github.io/DroneWatch/getting-started/)
- [configuration](https://bussler.github.io/DroneWatch/configuration/)
- [training-pipeline](https://bussler.github.io/DroneWatch/training-pipeline/)
- [results](https://bussler.github.io/DroneWatch/results/)

Publish new documentation with
```
make docs-build
uv run mkdocs gh-deploy --clean
```
