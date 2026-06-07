# Configuration

DroneWatch uses a two-stage configuration flow:

1. OmegaConf composes YAML fragments selected by the root config file and its `defaults` list.
2. Pydantic validates the resolved result against typed models in `src/dronewatch/config/schema.py`.

This means the YAML structure is flexible enough for grouped presets, while the final config still has a strict schema.

## Root Config Files

The repository currently uses different root configs for different workflows:

- `configs/config.yaml` for normal PPO training
- `configs/debug.yaml` for the PPO smoke path
- `configs/evaluate.yaml` for standalone checkpoint evaluation
- `configs/random_policy.yaml` for the random baseline
- `configs/tune_ppo.yaml` for Ray Tune sweeps

## How Composition Works

The loader functions in `src/dronewatch/config/loader.py` all follow the same pattern:

1. Load the root config
2. Read the `defaults` list
3. Resolve each config group from `configs/<group>/<name>.yaml`
4. Merge the selected group files
5. Apply CLI `key=value` overrides
6. Resolve interpolations
7. Validate the final payload with Pydantic

The main entrypoints are:

- `load_config()` for PPO training
- `load_tune_config()` for Tune
- `load_evaluation_config()` for checkpoint evaluation
- `load_random_policy_config()` for the random baseline

## Group Overrides vs Field Overrides

This distinction is the most important part of the config system.

### Group override

A group override replaces one preset from a config group.

```bash
uv run python -m dronewatch.training.train_ppo \
  --config configs/config.yaml \
  model=ppo_feedforward
```

Here `model=ppo_feedforward` switches the selected file under `configs/model/`.

### Field override

A field override changes a specific value inside the composed config.

```bash
uv run python -m dronewatch.training.train_ppo \
  --config configs/config.yaml \
  training.stop.iterations=1 \
  training.ppo.minibatch_size=64
```

Here the selected config groups stay the same, but the final values change before validation.

## Current Training Default

At the time of writing, `configs/config.yaml` selects:

- `env: swarm_search_v2`
- `model: ppo_lstm`
- `training: local_ppo`
- `logging: mlflow_local`
- `rendering: default`

That makes the obstacle-aware LSTM path the main training default.

## Where Resolved Configs Go

Each run writes a resolved YAML snapshot named `resolved_config.yaml` beside the artifact directory for that run. This is the best source of truth when you need to know what actually ran after group selection and CLI overrides were applied.

## What the Schema Owns

The typed models in `src/dronewatch/config/schema.py` define:

- project metadata and artifact locations
- environment and simulator settings
- observation limits
- reward weights
- PPO hyperparameters
- Ray worker allocation
- evaluation settings
- MLflow logging settings
- Tune search-space metadata

When a YAML field does not match this schema, validation fails before training or evaluation starts.
