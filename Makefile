.PHONY: help sync install develop-rust rollout-rust rollout-random render-random train-ppo tune-ppo tune-ppo-smoke evaluate-ppo render-ppo ppo-smoke mlflow-up mlflow-down mlflow-ui test-rust test-python test clean docker-build

help:
	@echo "Available targets:"
	@echo "  sync          Install Python dependencies with uv"
	@echo "  install       Sync Python deps and build the Rust extension in-place"
	@echo "  develop-rust  Build/install the PyO3 extension with maturin"
	@echo "  rollout-rust   Run a deterministic scripted Rust simulation rollout"
	@echo "  rollout-random Run a random policy rollout and write a JSON report"
	@echo "  render-random  Run a random policy rollout and write a report plus GIF"
	@echo "  train-ppo      Train shared-policy PPO locally"
	@echo "  tune-ppo       Run a local Ray Tune PPO hyperparameter search"
	@echo "  tune-ppo-smoke Run a tiny Ray Tune PPO smoke check"
	@echo "  evaluate-ppo   Evaluate PPO checkpoint; pass CHECKPOINT=path"
	@echo "  ppo-smoke      Run one tiny PPO training/evaluation smoke check"
	@echo "  mlflow-up      Start the MLflow tracking UI with Docker Compose"
	@echo "  mlflow-down    Stop the Docker Compose MLflow service"
	@echo "  mlflow-ui      Start a local MLflow UI without Docker"
	@echo "  test-rust     Run Rust tests"
	@echo "  test-python   Run Python tests"
	@echo "  test          Run Rust and Python tests"
	@echo "  clean         Remove generated build/test artifacts"
	@echo "  docker-build  Build the Phase 0 validation image"

sync:
	uv sync --dev

install: sync develop-rust

develop-rust:
	uv run maturin develop -m rust/swarm_sim/Cargo.toml

rollout-rust:
	uv run python scripts/run_rust_rollout.py

rollout-random:
	uv run python -m scripts.random_policy --config configs/random_policy.yaml

render-random:
	uv run python -m scripts.random_policy --config configs/random_policy.yaml random_policy.render=true

train-ppo:
	uv run python -m dronewatch.training.train_ppo --config configs/config.yaml

tune-ppo:
	uv run python -m dronewatch.training.tune_ppo --config configs/tune_ppo.yaml

tune-ppo-smoke:
	uv run python -m dronewatch.training.tune_ppo --config configs/tune_ppo.yaml tune.num_samples=2 training.stop.iterations=1 training.evaluation.enabled=false logging.mlflow.enabled=false

evaluate-ppo:
ifndef CHECKPOINT
	$(error CHECKPOINT=path/to/checkpoint is required)
endif
	uv run python -m dronewatch.evaluation.evaluate --config configs/evaluate.yaml evaluation.checkpoint=$(CHECKPOINT)

ppo-smoke:
	uv run python -m dronewatch.training.train_ppo --config configs/debug.yaml

mlflow-up:
	docker compose up mlflow

mlflow-down:
	docker compose down

mlflow-ui:
	uv run mlflow ui --backend-store-uri outputs/mlruns

test-rust:
	cargo test --manifest-path rust/swarm_sim/Cargo.toml

test-python:
	uv run pytest tests/python

test: test-rust test-python

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov
	rm -rf build dist *.egg-info src/*.egg-info src/dronewatch.egg-info
	rm -rf target rust/swarm_sim/target

docker-build:
	docker build -t dronewatch:phase0 .
