# DroneWatch

DroneWatch is a MARL engineering showcase where 16 cooperative drones learn, via RLlib PPO and later MAPPO-style centralized critic training, to discover targets in a partially observable continuous 2D environment with obstacles, collisions, local sensing, and short-range communication.

The repository is currently implementing Phase 6 of the project plan: Ray Tune hyperparameter search on top of the completed Rust simulation core, Python multi-agent environment wrapper, RLlib PPO path, OmegaConf experiment configuration, and local MLflow tracking.

## Current Phase

Phase 6 establishes a local Ray Tune workflow for PPO hyperparameter search, best-trial reporting, stable checkpoint artifacts, and MLflow parent/child run tracking for sweeps.

Implemented in this slice:

- Python package: `dronewatch`
- Rust extension module: `swarm_sim`
- Python wrapper: `dronewatch.sim`
- Rust world reset and step loop
- 16 default agents with continuous 2D movement
- bounded 100x100 world and fixed 200-step horizon
- static targets and circular no-fly obstacles
- target discovery, collision counts, obstacle violation counts
- coverage grid tracking
- communication connectivity metrics
- deterministic scripted rollout
- RLlib `MultiAgentEnv` wrapper: `SwarmSearchEnv`
- fixed-size local observations for all agents
- cooperative team reward calculation
- random policy baseline
- random rollout JSON report and GIF rendering
- RLlib PPO configuration for shared-policy training
- feedforward and LSTM model presets
- local PPO training CLI
- checkpoint evaluation CLI
- OmegaConf config composition under `configs/`
- Pydantic config validation in `dronewatch.config`
- CLI `key=value` overrides for training, evaluation, random rollout, and rendering settings
- resolved config snapshots saved with run artifacts
- local MLflow tracking under `outputs/mlruns`
- training metrics logged to MLflow
- checkpoint evaluation reports logged to MLflow
- Docker Compose MLflow UI service
- Ray Tune PPO hyperparameter search CLI
- Ray-style Tune search-space configuration
- Tune search summary reports under `artifacts/reports/`
- Tune trial checkpoints under `artifacts/checkpoints/ppo/tune/`
- PPO checkpoint saving under `artifacts/checkpoints/ppo/`
- PPO evaluation reports under `artifacts/reports/`
- uv dependency management
- PyO3/maturin build path
- Rust and Python tests
- Makefile and Docker skeleton

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Rust toolchain with Cargo
- Docker, optional for container validation

## Local Setup

Install Python dependencies and build the Rust extension into the uv environment:

```bash
make install
```

This runs:

```bash
uv sync --dev
uv run maturin develop -m rust/swarm_sim/Cargo.toml
```

## Smoke Check

Verify that Python can import the package, call into Rust, run a scripted simulator rollout, and run the random policy baseline:

```bash
uv run python -c "import dronewatch; print(dronewatch.__version__)"
uv run python -c "from dronewatch.sim import rust_version; print(rust_version())"
make rollout-rust
make rollout-random
make render-random
make ppo-smoke
make tune-ppo-smoke
```

`make rollout-random` writes a random policy report to `artifacts/reports/random_policy_report.json`.
`make render-random` writes the same report plus `artifacts/gifs/random_policy_episode.gif`.
`make ppo-smoke` runs one tiny feedforward PPO iteration, saves a checkpoint, and writes `artifacts/reports/ppo_smoke_report.json`.
`make tune-ppo-smoke` runs two tiny Ray Tune PPO trials and writes `artifacts/reports/tune_search_results.json`.

## Configuration

DroneWatch scripts now load structured YAML configs and accept dotlist overrides:

```bash
uv run python -m dronewatch.training.train_ppo \
	--config configs/config.yaml \
	model=ppo_lstm \
	training.stop.iterations=1 \
	training.ppo.train_batch_size_per_learner=200 \
	training.ppo.minibatch_size=64
```

The root configs are:

- `configs/config.yaml` for local PPO training defaults.
- `configs/debug.yaml` for a one-iteration smoke path.
- `configs/evaluate.yaml` for standalone PPO checkpoint evaluation.
- `configs/random_policy.yaml` for standalone random-policy baseline runs.

Ray Tune search-space settings live in `configs/tune/ray_tune.yaml`, and the local tuning training preset lives in `configs/training/tune_ppo.yaml`.

Config groups live under `configs/env`, `configs/model`, `configs/training`, `configs/evaluation`, `configs/random_policy`, `configs/logging`, `configs/rendering`, and `configs/tune`. Training, standalone evaluation, and random-policy runs each load their own root config, so standalone random-policy overrides use `random_policy.*` and standalone checkpoint evaluation overrides use `evaluation.*`.

Each training run writes a resolved YAML snapshot named `resolved_config.yaml` beside its checkpoint artifacts. Standalone checkpoint evaluation writes the same snapshot beside its report artifact.

## PPO Training

Run a local feedforward PPO training job and evaluate the final checkpoint:

```bash
make train-ppo
```

For direct control over training settings:

```bash
uv run python -m dronewatch.training.train_ppo \
	--config configs/config.yaml \
	training.stop.iterations=10 \
	model=ppo_feedforward \
	training.checkpoint.frequency_iters=5 \
	training.evaluation.episodes=5 \
	training.evaluation.report_path=reports/ppo_eval_report.json
```

Run the LSTM PPO smoke path after the feedforward path is working:

```bash
uv run python -m dronewatch.training.train_ppo \
	--config configs/debug.yaml \
	model=ppo_lstm \
	training.checkpoint.directory=checkpoints/ppo/lstm_smoke \
	training.evaluation.episodes=1 \
	training.ray.num_env_runners=0
```

Evaluate a saved checkpoint:

```bash
make evaluate-ppo CHECKPOINT=artifacts/checkpoints/ppo/path-to-checkpoint
```

Or directly:

```bash
uv run python -m dronewatch.evaluation.evaluate \
	--config configs/evaluate.yaml \
	evaluation.checkpoint=artifacts/checkpoints/ppo/path-to-checkpoint \
	evaluation.render=true
```

PPO reports use the same task-metric schema as the random baseline, which makes it straightforward to compare target discovery rate, coverage ratio, collisions, obstacle violations, connectivity, and success rate.

## Ray Tune Search

Run a local PPO hyperparameter search:

```bash
make tune-ppo
```

Run the small plumbing check:

```bash
make tune-ppo-smoke
```

The sweep optimizes the training-time metric configured at `tune.metric`, currently `target_discovery_rate`. Search results are written to `artifacts/reports/tune_search_results.json`; trial checkpoints are written under `artifacts/checkpoints/ppo/tune/`. When training evaluation is enabled for the tuning preset, the best checkpoint is evaluated with the existing PPO evaluation path and written to `artifacts/reports/tune_best_trial_report.json`.

Search spaces use explicit Ray-style sampler specs:

```yaml
tune:
	metric: target_discovery_rate
	mode: max
	num_samples: 12
	search_space:
		training.ppo.lr:
			type: loguniform
			lower: 0.0001
			upper: 0.001
		training.ppo.entropy_coeff:
			type: choice
			values: [0.0, 0.005, 0.01, 0.02]
```

## MLflow Tracking

Training and standalone checkpoint evaluation log parameters, task metrics, resolved config artifacts, and evaluation report artifacts to MLflow by default. The local tracking store is:

```text
outputs/mlruns
```

Start the Docker Compose MLflow server and open `http://localhost:5000`:

```bash
make mlflow-up
```

Stop it with:

```bash
make mlflow-down
```

You can also run the MLflow UI directly without Docker:

```bash
make mlflow-ui
```

PPO checkpoints and GIFs remain local files under `artifacts/` by default rather than being copied into MLflow.

## Tests

Run the full Phase 2 test suite:

```bash
make test
```

Or run each side directly:

```bash
make test-rust
make test-python
```

## Docker

Build the Phase 0 validation image:

```bash
make docker-build
```

The image installs the uv-managed project, builds the PyO3 extension, and defaults to `make test`.

The Docker Compose MLflow service is independent from the validation image and reads runs from `outputs/mlruns`.
