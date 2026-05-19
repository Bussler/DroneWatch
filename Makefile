.PHONY: help sync install develop-rust rollout-rust rollout-random render-random train-ppo evaluate-ppo render-ppo ppo-smoke test-rust test-python test clean docker-build

help:
	@echo "Available targets:"
	@echo "  sync          Install Python dependencies with uv"
	@echo "  install       Sync Python deps and build the Rust extension in-place"
	@echo "  develop-rust  Build/install the PyO3 extension with maturin"
	@echo "  rollout-rust   Run a deterministic scripted Rust simulation rollout"
	@echo "  rollout-random Run a random policy rollout and write a JSON report"
	@echo "  render-random  Run a random policy rollout and write a report plus GIF"
	@echo "  train-ppo      Train shared-policy PPO locally"
	@echo "  evaluate-ppo   Evaluate PPO checkpoint; pass CHECKPOINT=path"
	@echo "  ppo-smoke      Run one tiny PPO training/evaluation smoke check"
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
	uv run python -m dronewatch.baselines.random_policy --config configs/random_policy.yaml random_policy.episodes=1 project.seed=42 random_policy.report_path=artifacts/reports/random_policy_report.json random_policy.render=false

render-random:
	uv run python -m dronewatch.baselines.random_policy --config configs/random_policy.yaml random_policy.episodes=1 project.seed=42 random_policy.report_path=artifacts/reports/random_policy_report.json random_policy.gif_path=artifacts/gifs/random_policy_episode.gif random_policy.render=true

train-ppo:
	uv run python -m dronewatch.training.train_ppo --config configs/config.yaml

evaluate-ppo:
ifndef CHECKPOINT
	$(error CHECKPOINT=path/to/checkpoint is required)
endif
	uv run python -m dronewatch.evaluation.evaluate --config configs/evaluate.yaml evaluation.checkpoint=$(CHECKPOINT) evaluation.render=true evaluation.gif_path=artifacts/gifs/ppo_eval_episode.gif

ppo-smoke:
	uv run python -m dronewatch.training.train_ppo --config configs/debug.yaml

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
