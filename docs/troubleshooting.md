# Troubleshooting

## I changed a config value, but the wrong preset still loaded

Check whether you intended a group override or a field override.

- Use `model=ppo_lstm` to select a different preset file from `configs/model/`
- Use `training.stop.iterations=1` to change a value inside the selected config

If in doubt, inspect the run's `resolved_config.yaml` artifact.

## Evaluation says `evaluation.checkpoint` must be set

Standalone evaluation requires a checkpoint path at runtime. Pass it on the CLI:

```bash
make evaluate-ppo CHECKPOINT=artifacts/checkpoints/ppo/path-to-checkpoint
```

Or run the module directly with:

```bash
uv run python -m dronewatch.evaluation.evaluate \
  --config configs/evaluate.yaml \
  evaluation.checkpoint=artifacts/checkpoints/ppo/path-to-checkpoint
```

## RLlib cannot find the environment

Environment registration must happen before RLlib builds or reloads the algorithm. In this repository that registration lives in `src/dronewatch/training/rllib_config.py`.

If you move training or evaluation code, keep `register_swarm_search_env()` on the path before algorithm creation or checkpoint loading.

## MLflow has no runs or missing metrics

MLflow is optional. Check the active logging config under `configs/logging/` and confirm `logging.mlflow.enabled` is still true.

If MLflow is disabled, the run may still succeed and still write local reports under `artifacts/reports/`.

## I need to know what actually ran

The best source of truth is the resolved config snapshot written with the run artifacts. Use that file instead of inferring behavior from the root YAML alone.

## Reward looks good, but the behavior still looks wrong

Open the JSON report and compare reward against:

- success rate
- discovered target count
- collision count
- obstacle violation count

In obstacle-heavy runs, reward can stay high even when collision or obstacle behavior is still poor.
