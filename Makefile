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
	uv run python -m dronewatch.baselines.random_policy --config configs/config.yaml baseline.random.episodes=1 project.seed=42 baseline.random.report_path=artifacts/reports/random_policy_report.json

render-random:
	uv run python -m dronewatch.baselines.random_policy --config configs/config.yaml baseline.random.episodes=1 project.seed=42 baseline.random.report_path=artifacts/reports/random_policy_report.json baseline.random.gif_path=artifacts/gifs/random_policy_episode.gif baseline.random.render=true

train-ppo:
	uv run python -m dronewatch.training.train_ppo --config configs/config.yaml training.stop.iterations=10 model=ppo_feedforward training.checkpoint.directory=artifacts/checkpoints/ppo training.checkpoint.frequency_iters=5 training.evaluation.episodes=5 training.evaluation.report_path=artifacts/reports/ppo_eval_report.json

evaluate-ppo:
ifndef CHECKPOINT
	$(error CHECKPOINT=path/to/checkpoint is required)
endif
	uv run python -m dronewatch.evaluation.evaluate --config configs/config.yaml evaluation.checkpoint=$(CHECKPOINT) evaluation.episodes=10 evaluation.report_path=artifacts/reports/ppo_eval_report.json evaluation.render=true evaluation.gif_path=artifacts/gifs/ppo_eval_episode.gif

ppo-smoke:
	uv run python -m dronewatch.training.train_ppo --config configs/debug.yaml training.stop.iterations=1 model=ppo_feedforward training.checkpoint.directory=artifacts/checkpoints/ppo/smoke training.checkpoint.frequency_iters=1 training.evaluation.episodes=1 training.evaluation.report_path=artifacts/reports/ppo_smoke_report.json training.ray.num_env_runners=0 training.ppo.train_batch_size_per_learner=200 training.ppo.minibatch_size=64 training.ppo.num_epochs=1

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
