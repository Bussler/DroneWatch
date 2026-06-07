# DroneWatch

DroneWatch is a multi-agent reinforcement learning project that combines a Rust simulator with a Python training and evaluation stack. The current implementation supports local setup, scripted and random rollouts, PPO training, checkpoint evaluation, Ray Tune sweeps, MLflow logging, and artifact generation.

The project trains multiple cooperative drones to discover targets in a partially observable continuous 2D environment with obstacles, collisions, local sensing, and short-range communication.


## Implemetation

- Rust simulator exposed through the `swarm_sim` extension and the `dronewatch.sim` wrapper.
- RLlib `MultiAgentEnv` wrapper in `SwarmSearchEnv`.
- Structured YAML configuration composed with OmegaConf and validated with Pydantic.
- Shared-policy PPO training and checkpoint evaluation.
- Random-policy and scripted rollout baselines.
- JSON reports, GIF rendering, and MLflow experiment logging.

