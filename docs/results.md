# Results

DroneWatch writes JSON reports that are designed to answer a practical question: did the trained policy actually solve the task, and did it do so cleanly?

## Read These Metrics First

When you inspect a checkpoint report, start with:

1. `success_rate`
2. `mean_discovered_target_count`
3. `mean_target_discovery_rate`
4. `mean_episode_length`

These metrics answer whether the agent reliably finishes the task and how quickly it does so.

Then inspect:

- `mean_collision_count`
- `mean_obstacle_violation_count`
- `mean_connectivity_ratio`
- `mean_coverage_ratio`

Finally, if total reward looks surprising, inspect the reward-term breakdown.

## Example: PPO Smoke Failure

`artifacts/reports/ppo_smoke_report.json` is a useful example of a run that is valid but not good.

It currently shows:

- `success_rate = 0.0`
- `mean_discovered_target_count = 3.0`
- `mean_target_discovery_rate = 0.375`
- `mean_episode_length = 200.0`
- `mean_collision_count = 900.0`

This is a good example of why a successful training command is not the same thing as a good policy.

## Example: Successful Trained Policy

`artifacts/reports/ppo_eval_report.json` is the main example of a good checkpoint.

It currently shows:

- `success_rate = 1.0`
- `mean_discovered_target_count = 8.0`
- `mean_target_discovery_rate = 1.0`
- `mean_episode_length = 17.4`
- `mean_obstacle_violation_count = 0.0`

The same report also shows non-zero collisions, which means a successful policy can still leave room for improvement in swarm coordination.

## Example: Obstacle Run Tradeoffs

`artifacts/reports/DroneWatchLSTMGeneralizationObstacles/iteration_0500.json` is the right example when you want to understand obstacle-heavy training.

It currently shows a policy that consistently succeeds:

- `success_rate = 1.0`
- `mean_discovered_target_count = 20.0`
- `mean_target_discovery_rate = 1.0`

But it also shows non-trivial constraint costs:

- `mean_collision_count = 9.0`
- `mean_obstacle_violation_count = 3.9`
- `mean_connectivity_ratio = 0.275`

The reward-term breakdown explains why reward can still look strong in this situation. Large positive terms from target discovery and success can dominate the negative terms from collisions and obstacle violations.

## Practical Interpretation Rules

Use these rules when comparing checkpoints:

1. Prefer checkpoints with higher `success_rate` before looking at reward.
2. If success is tied, prefer fewer collisions and obstacle violations.
3. Use reward to compare efficient policies, not to decide whether the task was solved.
4. Treat connectivity as a diagnostic metric in v1, not as a failure condition by itself.

## Where Reports and Runs Live

- JSON reports: `artifacts/reports/`
- GIFs: `artifacts/gifs/`
- Checkpoints: `artifacts/checkpoints/ppo/`
- MLflow runs: `outputs/mlruns/`
