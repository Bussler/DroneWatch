# DroneWatch Project and Implementation Plan

## 1. Project Summary

**DroneWatch** is a multi-agent reinforcement learning engineering showcase. The project builds a Rust-based cooperative drone swarm simulation and trains agents with Ray RLlib to discover targets in a partially observable continuous 2D world.

The project demonstrates:

- simulation engineering in Rust
- Python/Rust interoperability through PyO3 and maturin
- multi-agent reinforcement learning with Ray RLlib
- recurrent shared-policy PPO for partial observability
- later MAPPO-style centralized critic training
- structured experiment configuration with OmegaConf
- metric tracking with MLflow
- Dockerized training and evaluation workflows
- production-quality testing and documentation

The intended audience is ML engineering, research engineering, reinforcement learning, simulation, robotics, and swarm intelligence roles.

---

## 2. High-Level Goal

Build an end-to-end MARL platform where **16 homogeneous drones** cooperatively search for targets in a continuous 2D environment.

The environment contains:

- continuous 2D world coordinates
- simple drone kinematics
- local partial observations
- short-range communication
- static targets
- static obstacles/no-fly zones
- collision detection
- area coverage tracking
- fixed-length episodes capped at 200 steps

The primary objective is **target discovery**.

The secondary objective is **area coverage**.

Communication connectivity should be tracked as a metric in v1, but it should **not** be part of the reward objective in the MVP.

---

## 3. Architecture Overview

```text
OmegaConf Configs
      |
      v
Python Training / Evaluation Layer
      |
      |  Ray RLlib
      |  MLflow
      |  Gymnasium / RLlib MultiAgentEnv
      |  Reward calculation
      |  Observation construction
      |  Rendering
      v
PyO3 / maturin Boundary
      |
      v
Rust Simulation Core
      |
      |  World state
      |  Agent movement
      |  Target discovery
      |  Obstacles
      |  Collisions
      |  Coverage grid
      |  Communication graph metrics
      v
Simulation Metrics and Events
```

---

## 4. Core Design Decisions

### 4.1 Engineering Focus

This project is primarily an engineering-heavy MARL platform, not a pure research algorithm project.

Prioritize:

- correctness
- modularity
- testability
- reproducibility
- clean interfaces
- clear documentation
- usable scripts and commands

Avoid unnecessary complexity in early versions.

### 4.2 Simulation Scope

Use simple continuous 2D kinematics.

Do **not** implement:

- full drone physics
- 3D simulation
- motor-level control
- aerodynamics
- complex flight dynamics

### 4.3 RL Scope

The first training baseline should be:

- RLlib PPO
- parameter-shared policy
- homogeneous agents
- LSTM policy
- shared cooperative team reward

Later extensions may include:

- MAPPO-style centralized critic
- heterogeneous agents
- curriculum learning
- communication-aware reward shaping

---

## 5. Recommended Repository Structure

```text
DroneWatch/
  README.md
  PROJECT_PLAN.md
  pyproject.toml
  Cargo.toml
  Dockerfile
  docker-compose.yml
  Makefile

  rust/
    swarm_sim/
      Cargo.toml
      pyproject.toml
      src/
        lib.rs
        world.rs
        agent.rs
        target.rs
        obstacle.rs
        geometry.rs
        metrics.rs
        config.rs
      tests/
        test_world.rs
        test_geometry.rs
        test_collisions.rs

  src/
    dronewatch/
      __init__.py

      envs/
        __init__.py
        swarm_search_env.py
        observation_builder.py
        reward.py
        spaces.py

      training/
        __init__.py
        train_ppo.py
        tune_ppo.py
        train_mappo.py

      evaluation/
        __init__.py
        evaluate.py
        scenarios.py

      rendering/
        __init__.py
        render_episode.py
        gif_writer.py

      logging/
        __init__.py
        mlflow_logger.py

      config/
        __init__.py
        loader.py
        schema.py

      baselines/
        __init__.py
        random_policy.py

      utils/
        __init__.py
        seeding.py
        paths.py

  configs/
    config.yaml

    env/
      swarm_search_v1.yaml
      swarm_search_eval.yaml

    model/
      ppo_lstm.yaml
      ppo_feedforward_debug.yaml
      mappo_lstm.yaml

    training/
      local_ppo.yaml
      debug.yaml
      ray_tune.yaml

    logging/
      mlflow_local.yaml

    evaluation/
      eval_default.yaml
      eval_hard.yaml

  scripts/
    train.sh
    evaluate.sh
    render_random_policy.sh
    render_checkpoint.sh

  tests/
    python/
      test_env_reset.py
      test_env_step.py
      test_observations.py
      test_rewards.py
      test_config_loading.py
      test_random_policy.py

  docs/
    architecture.md
    environment.md
    training.md
    experiments.md
    results.md

  outputs/
    .gitkeep

  artifacts/
    checkpoints/
      .gitkeep
    gifs/
      .gitkeep
    reports/
      .gitkeep
```

The structure can evolve, but keep the project as a single repository.

---

## 6. Environment Specification

### 6.1 Environment Name

Use the environment name:

```text
SwarmSearch2D
```

### 6.2 World

The world is a bounded continuous 2D area.

Default values:

```yaml
world:
  width: 100.0
  height: 100.0
  dt: 1.0
```

Coordinates:

```text
x in [0, width]
y in [0, height]
```

### 6.3 Episode

Episodes are fixed horizon.

Default:

```yaml
max_episode_steps: 200
```

An episode ends when:

- all targets are discovered, or
- the maximum episode length is reached

Use Gymnasium/RLlib-compatible `terminated` and `truncated` semantics.

### 6.4 Agents

MVP default:

```yaml
simulation:
  agents:
    count: 16
```

Agents are homogeneous in v1.

Each agent should have:

- unique ID
- 2D position
- optional 2D velocity
- maximum speed
- collision radius
- sensing radius
- communication radius
- active/alive flag if useful later

### 6.5 Targets

Targets are static in v1.

Each target has:

- 2D position
- discovered flag
- discovery radius
- optional value/weight later

### 6.6 Obstacles / No-Fly Zones

Use simple obstacle geometry:

- circles for v1
- optionally axis-aligned rectangles later

Obstacles should affect:

- collision checking
- no-fly violation metrics
- observations
- rendering

### 6.7 Agent Dynamics

Use simple action-controlled movement.

Preferred MVP action:

```text
action = [dx, dy]
```

Where:

- each component is continuous in `[-1.0, 1.0]`
- the vector determines movement direction
- movement speed is capped
- world boundaries are enforced

Avoid complex acceleration dynamics in v1.

---

## 7. Observation Design

Partial observability is a core feature.

Each agent only observes local information.

Observations must be fixed-size vectors compatible with RLlib and LSTM policies.

Recommended observation components:

- normalized own position
- own velocity if modeled
- normalized timestep fraction
- nearest visible agents within sensing radius
- nearest visible targets within sensing radius
- nearest visible obstacles
- short-range communication summary

Use zero-padding and masks for variable numbers of visible entities.

Recommended config:

```yaml
observation:
  max_visible_agents: 5
  max_visible_targets: 5
  max_visible_obstacles: 5
  include_communication_summary: true
```

Communication summary may include:

- normalized number of communication neighbors
- mean relative x/y position of communication neighbors
- mean relative velocity of communication neighbors, if velocity exists

Do not expose global state to the actor policy in the MVP.

Global state may be added later for MAPPO-style centralized critic training.

---

## 8. Reward Design

Use a cooperative team reward in the MVP.

All agents receive the same reward at each step.

Recommended reward formula:

```text
reward =
    target_discovery_reward
  + coverage_reward
  - collision_penalty
  - obstacle_penalty
  - step_penalty
```

Recommended initial reward values:

```yaml
reward:
  target_discovered: 5.0
  new_coverage_cell: 0.02
  agent_collision: -0.25
  obstacle_collision: -0.5
  step_penalty: -0.001
```

Reward calculation should live in Python unless implementation pressure suggests otherwise.

Rust should expose the raw simulation events and metrics needed for reward calculation.

Avoid excessive reward shaping early.

Always compare reward to task metrics during debugging.

---

## 9. Metrics

Metrics are first-class project artifacts.

Do not rely only on episode reward.

### 9.1 Required Environment Metrics

Track at least:

- target discovery rate
- number of discovered targets
- coverage ratio
- collision count
- obstacle violation count
- communication graph connectivity ratio
- average communication neighbors
- episode length
- success rate

### 9.2 Communication Connectivity Metric

For each step:

1. Create a graph with drones as nodes.
2. Add an edge when two drones are within communication range.
3. Compute the size of the largest connected component.

Recommended metric:

```text
connectivity_ratio = largest_connected_component_size / num_agents
```

This is a metric in v1, not a reward term.

### 9.3 Training Metrics

Log RLlib metrics such as:

- episode reward mean
- policy loss
- value loss
- entropy
- KL divergence
- environment steps
- learner throughput

### 9.4 Evaluation Metrics

Evaluation reports should include:

- mean reward
- mean target discovery rate
- mean discovered target count
- mean coverage ratio
- mean collision count
- mean obstacle violation count
- mean connectivity ratio
- success rate
- mean episode length

---

## 10. Rust Simulation Core

### 10.1 Rust Responsibilities

Rust should handle:

- world state
- world reset
- world step
- agent movement
- boundary handling
- target discovery events
- obstacle/no-fly collision detection
- agent-agent collision detection
- coverage grid updates
- communication graph calculations
- raw simulation metrics

### 10.2 Suggested Rust Modules

```text
geometry.rs   distance functions, circles, bounds, collision helpers
agent.rs      drone state and movement logic
target.rs     target state and discovery logic
obstacle.rs   obstacle representation and collision checks
world.rs      full simulation state, reset, step
metrics.rs    per-step and per-episode metrics
config.rs     Rust-side config structs if needed
lib.rs        PyO3 bindings and public exports
```

### 10.3 Rust/Python Boundary

Expose Rust through PyO3/maturin.

Keep the API coarse-grained.

Preferred calls:

```text
sim.reset(...)
sim.step(actions)
sim.get_state()
sim.get_metrics()
```

Avoid excessive per-agent calls across the Rust/Python boundary.

---

## 11. Python Layer

### 11.1 Python Responsibilities

Python should handle:

- RLlib `MultiAgentEnv`
- observation construction
- reward calculation
- terminated/truncated dictionaries
- config loading
- training scripts
- evaluation scripts
- rendering
- MLflow logging

### 11.2 RLlib Environment

Implement a class similar to:

```python
class SwarmSearchEnv(MultiAgentEnv):
    def reset(self, *, seed=None, options=None):
        ...

    def step(self, action_dict):
        ...
```

The step method should return:

```text
observations: dict[agent_id, observation]
rewards: dict[agent_id, reward]
terminateds: dict[agent_id, bool]
truncateds: dict[agent_id, bool]
infos: dict[agent_id, dict]
```

Include:

```python
terminateds["__all__"] = episode_done
truncateds["__all__"] = episode_truncated
```

---

## 12. RLlib Training Plan

### 12.1 MVP Algorithm

Use RLlib PPO as the primary algorithm.

Use parameter sharing:

```text
agent_0  -> shared_policy
agent_1  -> shared_policy
...
agent_15 -> shared_policy
```

Policy mapping:

```python
def policy_mapping_fn(agent_id, *args, **kwargs):
    return "shared_policy"
```

### 12.2 LSTM

Use LSTM for the main training configuration because the environment is partially observable.

However, implement this progression:

1. random policy
2. feedforward PPO smoke test
3. LSTM PPO
4. Ray Tune sweep
5. MAPPO-style centralized critic extension

This reduces debugging complexity.

### 12.3 Baselines

Implement and maintain:

- random policy baseline
- shared-policy PPO baseline

The random policy baseline should generate the same evaluation report format as PPO.

### 12.4 MAPPO Extension

MAPPO-style training is a later extension.

Preferred design:

- actor receives local observation
- critic receives global state
- execution remains decentralized

Do not block the MVP on MAPPO.

---

## 13. Configuration Plan

Use OmegaConf YAML configuration.

Config groups:

- environment
- model
- training
- logging
- evaluation

### 13.1 Root Config

```yaml
# configs/config.yaml
defaults:
  - env: swarm_search_v1
  - model: ppo_lstm
  - training: local_ppo
  - logging: mlflow_local
  - evaluation: eval_default

project:
  name: DroneWatch
  seed: 133742

runtime:
  output_dir: outputs
  artifact_dir: artifacts
```

### 13.2 Environment Config

```yaml
# configs/env/swarm_search_v1.yaml
env:
  name: SwarmSearch2D

  simulation:
    max_episode_steps: 200

    world:
      width: 100.0
      height: 100.0
      dt: 1.0

    agents:
      count: 16
      max_speed: 2.0
      collision_radius: 0.75
      sensing_radius: 15.0
      communication_radius: 20.0

    targets:
      count: 20
      discovery_radius: 2.0

    obstacles:
      count: 8
      min_radius: 2.0
      max_radius: 6.0

    coverage:
      grid_width: 50
      grid_height: 50
      sensing_radius: 10.0

  observation:
    max_visible_agents: 5
    max_visible_targets: 5
    max_visible_obstacles: 5
    include_communication_summary: true

  reward:
    target_discovered: 5.0
    new_coverage_cell: 0.02
    agent_collision: -0.25
    obstacle_collision: -0.5
    step_penalty: -0.001
```

### 13.3 PPO LSTM Config

```yaml
# configs/model/ppo_lstm.yaml
model:
  kind: lstm
  network:
    use_lstm: true
    lstm_cell_size: 128
    max_seq_len: 20
    fcnet_hiddens: [256, 256]
    activation: tanh

  ppo:
    gamma: 0.99
    lambda_: 0.95
    lr: 0.0003
    clip_param: 0.2
    entropy_coeff: 0.01
    vf_loss_coeff: 1.0
    train_batch_size: 8192
    minibatch_size: 1024
    num_epochs: 10
```

### 13.4 Training Config

```yaml
# configs/training/local_ppo.yaml
training:
  stop:
    iterations: 10

  ray:
    num_env_runners: 4
    num_envs_per_env_runner: 1
    num_learners: 0
    num_gpus_per_learner: 0

  checkpoint:
    directory: artifacts/checkpoints/ppo
    frequency_iters: 25

  evaluation:
    interval_iters: 10
    episodes: 20
```

### 13.5 MLflow Config

```yaml
# configs/logging/mlflow_local.yaml
logging:
  mlflow:
    enabled: true
    tracking_uri: file:./outputs/mlruns
    experiment_name: dronewatch-swarm-search-ppo
    log_system_metrics: false

  console:
    enabled: true
```

---

## 14. MLflow Plan

MLflow should track training and evaluation metrics.

Do not store large checkpoints or GIFs in MLflow by default.

### 14.1 Log Parameters

Log:

- algorithm name
- number of agents
- episode length
- world size
- reward weights
- model settings
- training settings
- seed

### 14.2 Log Metrics

Log:

- episode reward mean
- target discovery rate
- discovered target count
- coverage ratio
- collision count
- obstacle violation count
- communication connectivity ratio
- average communication neighbors
- policy loss
- value loss
- entropy
- KL divergence
- environment steps

### 14.3 Log Artifacts

Log:

- resolved config YAML
- evaluation summary/report
- small summary plots if useful

### 14.4 Store Locally Outside MLflow

Store:

```text
artifacts/checkpoints/
artifacts/gifs/
artifacts/reports/
```

---

## 15. Evaluation Plan

Training and evaluation scenarios should be distinct.

### 15.1 Training Scenarios

Training maps may randomize:

- agent spawn positions
- target positions
- obstacle positions
- target density
- obstacle density

### 15.2 Evaluation Scenarios

Use fixed or reproducible scenario sets.

Recommended configs:

- `eval_default`
- `eval_hard`

### 15.3 Evaluation Report

Save reports under:

```text
artifacts/reports/
```

Recommended report shape:

```json
{
  "checkpoint": "...",
  "num_episodes": 100,
  "mean_reward": 0.0,
  "mean_target_discovery_rate": 0.0,
  "mean_discovered_target_count": 0.0,
  "mean_coverage_ratio": 0.0,
  "mean_collision_count": 0.0,
  "mean_obstacle_violation_count": 0.0,
  "mean_connectivity_ratio": 0.0,
  "success_rate": 0.0,
  "mean_episode_length": 0.0
}
```

---

## 16. Rendering Plan

Use matplotlib-based rendering.

The renderer should show:

- world bounds
- drone positions
- targets
- discovered targets
- obstacles/no-fly zones
- optional sensing radius
- optional communication edges
- current step
- discovered target count

Store GIFs under:

```text
artifacts/gifs/
```

Recommended GIFs:

- `random_policy_episode.gif`
- `ppo_checkpoint_episode.gif`
- `ppo_final_episode.gif`

The visual demo should be simple and readable.

---

## 17. Docker Plan

Docker should support local reproducible training and evaluation.

The image should include:

- Python
- Rust toolchain
- maturin
- Ray/RLlib
- PyTorch
- MLflow
- OmegaConf
- Gymnasium
- matplotlib
- local project package

Recommended commands:

```bash
make docker-build
make docker-train
make docker-evaluate
```

Use mounted volumes for:

```text
outputs/
artifacts/
```

GPU support is optional and not required for MVP.

---

## 18. Testing Plan

Testing is important because the project should look production-quality.

### 18.1 Rust Tests

Test:

- geometry utilities
- distance calculations
- world boundary handling
- agent movement
- speed clipping
- target discovery
- obstacle collision
- agent-agent collision
- coverage grid updates
- communication graph metrics
- world reset
- world step invariants

Important invariants:

- coverage ratio is always between 0 and 1
- connectivity ratio is always between 0 and 1
- discovered target count never decreases
- collision count is non-negative
- agent positions never become NaN
- world step increments timestep exactly once

### 18.2 Python Tests

Test:

- config loading
- environment reset
- environment step
- observation shape
- observation finite values
- action space validity
- reward calculation
- terminated/truncated semantics
- random policy rollout
- evaluation report generation
- renderer smoke test if practical

### 18.3 Integration Tests

Add lightweight integration tests:

- one full random episode
- one tiny PPO smoke test
- one evaluation report generation test

Avoid slow tests in the default test suite.

---

## 19. Suggested Makefile Commands

The repository should eventually support commands similar to:

```makefile
install:
	pip install -e ".[dev]"
	maturin develop -m rust/swarm_sim/Cargo.toml

test:
	cargo test --manifest-path rust/swarm_sim/Cargo.toml
	pytest tests/python

train:
	python -m dronewatch.training.train_ppo --config configs/config.yaml

tune:
	python -m dronewatch.training.tune_ppo --config configs/config.yaml

evaluate:
  python -m dronewatch.evaluation.evaluate --config configs/evaluate.yaml

render-random:
  python -m dronewatch.baselines.random_policy --config configs/random_policy.yaml random_policy.render=true

docker-build:
	docker build -t dronewatch:latest .

docker-train:
	docker run --rm -it \
		-v $$(pwd)/outputs:/app/outputs \
		-v $$(pwd)/artifacts:/app/artifacts \
		dronewatch:latest make train
```

Adapt commands as implementation details evolve.

---

## 20. Implementation Milestones

### Phase 0: Project Skeleton

Estimated time: 1-2 weeks

Goals:

- create Python package structure
- create Rust crate structure
- configure PyO3/maturin
- add Makefile
- add basic Dockerfile skeleton
- add initial README
- add placeholder docs

Deliverables:

- repository installs locally
- Rust module can be imported from Python
- trivial Rust function callable from Python
- `make test` works with placeholder tests

---

### Phase 1: Rust Simulation Core

Estimated time: 3-5 weeks

Goals:

- implement geometry helpers
- implement world state
- implement drone agents
- implement static targets
- implement circular obstacles
- implement continuous movement
- implement boundary handling
- implement collision detection
- implement target discovery
- implement coverage grid
- implement communication graph metrics

Deliverables:

- Rust world can reset and step
- Rust tests cover core primitives
- Python can call Rust reset/step
- random scripted simulation runs for 200 steps

Success criteria:

- random simulation produces discovered target count
- coverage ratio is computed
- collision count is computed
- obstacle violations are computed
- connectivity ratio is computed

---

### Phase 2: Python Environment Wrapper

Estimated time: 2-4 weeks

Goals:

- implement RLlib `MultiAgentEnv`
- implement observation builder
- implement reward calculator
- implement action conversion
- propagate metrics into infos
- implement random policy baseline

Deliverables:

- random policy rollout works
- fixed-size observations returned for all agents
- reward dict is valid
- terminated/truncated semantics are correct
- Python unit tests pass

Success criteria:

```bash
make render-random
```

produces:

```text
artifacts/gifs/random_policy_episode.gif
artifacts/reports/random_policy_report.json
```

---

### Phase 3: PPO Training with RLlib

Estimated time: 3-5 weeks

Goals:

- register RLlib environment
- implement shared policy mapping
- implement feedforward PPO smoke test
- implement LSTM PPO config
- save checkpoints
- generate evaluation report

Deliverables:

- PPO trains without crashing
- LSTM PPO runs
- checkpoints are saved
- evaluation script can evaluate checkpoints

Success criteria:

PPO should outperform random policy on:

- target discovery rate
- coverage ratio

Ideally PPO should also reduce:

- collision count
- obstacle violation count

---

### Phase 4: OmegaConf Experiment Structure

Estimated time: 1-2 weeks

Goals:

- add config groups
- add config loader
- add debug config
- add local training config
- add eval configs
- save resolved configs per run

Deliverables:

- scripts load from YAML configs
- debug runs are fast
- constants are no longer hard-coded unnecessarily

Success criteria:

```bash
python -m dronewatch.training.train_ppo --config configs/config.yaml
```

starts a configured experiment.

---

### Phase 5: MLflow Logging

Estimated time: 1-2 weeks

Goals:

- add local MLflow tracking
- log training metrics
- log evaluation metrics
- log resolved config artifact
- log evaluation report artifact

Deliverables:

- MLflow experiment directory under `outputs/mlruns`
- MLflow UI can compare runs
- metrics include task-specific environment metrics

Success criteria:

```bash
mlflow ui --backend-store-uri outputs/mlruns
```

shows DroneWatch experiments.

---

### Phase 6: Ray Tune Hyperparameter Search

Estimated time: 2-3 weeks

Goals:

- implement `tune_ppo.py`
- define configurable search space
- run search over PPO hyperparameters
- log results with MLflow
- generate best-trial report

Suggested search parameters:

- learning rate
- entropy coefficient
- train batch size
- LSTM cell size
- reward weights, optional

Deliverables:

- Ray Tune config
- reproducible tuning script
- best checkpoint/evaluation summary

---

### Phase 7: Dockerization

Estimated time: 1-2 weeks

Goals:

- implement Dockerfile
- optionally implement docker-compose
- support mounted outputs/artifacts
- document Docker commands

Deliverables:

- `make docker-build`
- `make docker-train`
- `make docker-evaluate`

Success criteria:

A debug training job can run inside Docker.

---

### Phase 8: Documentation and Polish

Estimated time: 2-4 weeks

Goals:

- polish README
- add architecture docs
- add environment docs
- add training docs
- add experiment docs
- add results docs
- include sample metrics and GIFs

Deliverables:

- understandable README
- documented commands
- sample results
- roadmap

Success criteria:

A reviewer can understand the project within 5 minutes and see the engineering depth within 15 minutes.

---

## 21. Later Extensions

### 21.1 MAPPO-Style Centralized Critic

Implement centralized training with decentralized execution.

Design:

- actor receives local observation
- critic receives global state
- shared actor policy for homogeneous drones

Global state may include:

- all drone positions
- all target discovery states
- obstacle summaries
- timestep

### 21.2 Heterogeneous Agents

Possible roles:

- scout drones with larger sensing radius
- relay drones with larger communication radius
- standard drones

Use this to study role specialization.

### 21.3 Curriculum Learning

Possible curriculum:

1. no obstacles, dense targets
2. few obstacles, dense targets
3. more obstacles
4. sparse targets
5. larger world

### 21.4 Communication-Aware Reward

Only after connectivity is already tracked as a metric, experiment with reward terms for:

- local communication density
- graph connectivity
- largest connected component size

---

## 22. Definition of Done: MVP

The MVP is complete when:

- Rust simulator can run 16 drones in a 200-step episode
- Python can call the simulator through PyO3/maturin
- RLlib multi-agent environment works
- random policy baseline can be evaluated
- random policy GIF can be generated
- shared-policy PPO with LSTM trains successfully
- PPO outperforms random policy on target discovery rate
- OmegaConf controls environment, model, training, logging, and evaluation
- MLflow logs key training and evaluation metrics
- evaluation reports are saved locally
- Docker can run a debug training job
- Rust and Python unit tests cover the core system
- README explains how to install, train, evaluate, render, and inspect results

---

## 23. Definition of Done: Strong Portfolio Version

The strong portfolio version is complete when:

- Ray Tune hyperparameter sweeps are implemented
- default and hard evaluation scenarios exist
- results compare random policy vs PPO
- sample GIFs are included in the README
- MLflow screenshots or summarized metrics are documented
- architecture and training documentation are polished
- Docker workflow is documented and tested
- MAPPO-style centralized critic is implemented or clearly described as future work
- limitations and future work are stated honestly
