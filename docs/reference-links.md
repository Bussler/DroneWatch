# Reference Links

These are the main implementation anchors for the current documentation.

## Setup and workflows

- `README.md`
- `Makefile`
- `pyproject.toml`

## Configuration system

- `src/dronewatch/config/loader.py`
- `src/dronewatch/config/schema.py`
- `configs/config.yaml`
- `configs/debug.yaml`
- `configs/evaluate.yaml`
- `configs/random_policy.yaml`
- `configs/tune_ppo.yaml`

## Training and evaluation

- `src/dronewatch/training/train_ppo.py`
- `src/dronewatch/training/rllib_config.py`
- `src/dronewatch/training/callbacks.py`
- `src/dronewatch/envs/swarm_search_env.py`
- `src/dronewatch/envs/reward.py`
- `src/dronewatch/evaluation/evaluate.py`
- `src/dronewatch/evaluation/reporting.py`
- `src/dronewatch/logging/mlflow_logger.py`

## Example artifacts

- `artifacts/reports/ppo_smoke_report.json`
- `artifacts/reports/ppo_eval_report.json`
- `artifacts/reports/DroneWatchLSTMGeneralizationObstacles/iteration_0500.json`
- `outputs/mlruns/`
