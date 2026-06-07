# Training Pipeline

This page explains the Python-side control flow for PPO training and checkpoint evaluation.

## Pipeline Overview

```text
CLI
  -> config loader
  -> typed config object
  -> RLlib PPOConfig builder
  -> RLlib env registration
  -> SwarmSearchEnv reset/step loop
  -> reward attribution and task metrics
  -> checkpoints, JSON reports, MLflow logs
```

## Training Entry Point

The main training entrypoint is `src/dronewatch/training/train_ppo.py`.

`main()` parses `--config` plus CLI overrides, then calls `load_config()`. The resulting `DroneWatchConfig` is passed to `train_ppo()`.

Inside `train_ppo()` the important steps are:

1. Create the run artifact directory
2. Save `resolved_config.yaml`
3. Start the MLflow run if logging is enabled
4. Build the PPO algorithm from `build_ppo_config()`
5. Train for `training.stop.iterations`
6. Save checkpoints at `training.checkpoint.frequency_iters`
7. Optionally run periodic evaluation and write JSON reports
8. Save the final checkpoint and return a compact run summary

## RLlib Configuration

`src/dronewatch/training/rllib_config.py` owns the RLlib-specific setup.

Key responsibilities:

- `register_swarm_search_env()` registers the RLlib environment name
- `shared_policy_mapping_fn()` maps every agent to the same PPO policy
- `build_ppo_config()` turns the typed config models into an RLlib `PPOConfig`

The environment registration step matters because checkpoint loading and fresh algorithm creation both depend on the registered environment name.

## Environment Wrapper

`src/dronewatch/envs/swarm_search_env.py` is the Python wrapper around the Rust simulator.

For the Rust-side simulator API and module structure, see the Rust Environment page.

Important behavior:

- `reset()` resets the simulator, rebuilds fixed-size observations, and returns initial metrics in `infos`
- `step()` converts RLlib's action dictionary into ordered simulator actions
- reward terms are split between shared team reward and locally attributed agent reward
- final `infos` include simulator metrics plus episode reward breakdown

This is the main Python orchetration layer between RLlib and the Rust simulation state.

## Reward Logic

`src/dronewatch/envs/reward.py` defines the reward surface.

The current reward terms are:

- `target_discovery`
- `coverage`
- `agent_collision`
- `obstacle_collision`
- `step_penalty`
- `remaining_targets`
- `success_bonus`
- `visible_target_approach`

Some terms are shared across the team, while others are attributed to specific agents. That split is important when you interpret episode rewards and when you compare total reward against collisions or obstacle violations.

## Training Metrics

`src/dronewatch/training/callbacks.py` pushes task metrics into RLlib at episode end. These include:

- target discovery rate
- discovered target count
- coverage ratio
- collision count
- obstacle violation count
- connectivity ratio
- average communication neighbors
- success rate

These metrics are more informative than reward alone when you are deciding whether a policy is actually improving.

## Evaluation Path

`src/dronewatch/evaluation/evaluate.py` provides two main layers:

- `evaluate_checkpoint()` loads an algorithm from a saved checkpoint
- `evaluate_algorithm()` runs fresh episodes, optionally renders GIFs, and returns a structured report

The report format itself is defined in `src/dronewatch/evaluation/reporting.py` through:

- `episode_summary()`
- `aggregate_report()`
- `write_json_report()`

That report schema is shared across PPO evaluation and the random baseline so the outputs stay comparable.

## MLflow Integration

`src/dronewatch/logging/mlflow_logger.py` is intentionally small and explicit.

It handles:

- starting runs and nested child runs
- flattening config parameters
- logging numeric metrics
- logging evaluation summary metrics
- logging resolved configs and reports as artifacts

The local store is `outputs/mlruns/` unless overridden.
