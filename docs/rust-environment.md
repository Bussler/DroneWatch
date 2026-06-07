# Rust Environment

This page describes the Rust simulator that sits under the Python training stack.

In DroneWatch, the Rust side is not a Gym or RLlib environment by itself. It is the simulation core that owns the world state, applies agent movement, updates targets and coverage, computes raw metrics, and exposes a coarse-grained API to the Python training stack through PyO3.

## Where the Rust Boundary Lives

The Python extension module is defined in `rust/swarm_sim/src/lib.rs`.

The main Python-facing type is `SwarmWorld` (`src/dronewatch/sim.py`), which wraps the internal `World` struct from `rust/swarm_sim/src/world.rs`.

Python owns observations, rewards, RLlib integration, rendering, and MLflow logging. Rust owns the mutable simulation state and the raw episode metrics that those higher layers consume.

## Main Functions

The Rust environment exposes a small set of coarse-grained operations.

### `new(seed=None, config=None)`

Constructs a `SwarmWorld` from an optional seed and optional JSON config payload.

Important behavior:

- parses the JSON payload into `SimulationConfig`
- validates the configuration
- creates a `World`
- immediately resets the world with the resolved seed

### `reset(seed=None)`

Starts a fresh episode and returns the initial simulator metrics.

Inside `World::reset()` the simulator:

- clears timestep and cumulative counters
- samples a deterministic scenario from the seed
- repopulates agents, targets, and obstacles
- resets the coverage grid
- marks the initial covered cells from starting agent positions
- updates target discovery if an agent starts inside discovery radius
- recomputes communication graph metrics

The return value is a metrics dictionary, not a full observation. Python uses that output together with `state()` to build RLlib observations.

### `step(actions)`

Advances the simulation by exactly one timestep.

This is the main function used during training and evaluation.  
See [Input](#what-goes-into-step) and [Output](#what-comes-out-of-step) of the step function.

### `state()`

Returns a serializable snapshot of the current world state:

- `timestep`
- `agents` with `id`, `position`, and `velocity`
- `targets` with `id`, `position`, and `discovered`
- `obstacles` with `id`, `position`, and `radius`

### `metrics()`

Returns cumulative metrics for the current episode, including progress, collisions, coverage, connectivity, and done flags.

### `is_done()`

Returns `true` when either all targets are discovered or the episode horizon has been reached.

## How the Rust Code Is Structured

The simulator is deliberately split into small modules with clear ownership.

### `lib.rs`

Defines the PyO3 module, the `SwarmWorld` wrapper, config parsing from JSON, and Python serialization of state, events, and metrics.

### `world.rs`

Owns the mutable episode state and the main orchestration logic for `new`, `reset`, `step`, `metrics`, and `is_done`.

### `config.rs`

Defines `SimulationConfig` and its nested settings for the world, agents, targets, obstacles, and coverage grid. This file also validates that the Rust simulator can run with the supplied values.

### `metrics.rs`

Defines:

- `StepEvents` for per-step event counts
- `SimulationMetrics` for cumulative episode metrics
- `EventCollector` for pure calculations such as collision counts and communication graph metrics

### Supporting modules

- `agent.rs`: agent state
- `target.rs`: target state and discovery flag
- `obstacle.rs`: circular no-fly obstacles
- `geometry.rs`: 2D vector math and overlap helpers
- `coverage.rs`: coverage-grid bookkeeping
- `scenario.rs`: deterministic scenario generation from a seed

## What Goes Into `step`

At the Python boundary, `SwarmWorld.step()` receives a list of actions shaped like this:

```text
[(dx, dy), (dx, dy), ...]
```

There must be exactly one action for each active agent/ drone.

The Python wrapper `dronewatch.sim.SwarmSimulation.step()` converts Python sequences into tuples of floats, and Rust converts those tuples into `Vec2` values.

Inside `World::step()` the simulator validates that:

- the episode is not already done
- the number of actions matches the number of agents
- every action is finite

Each action is then processed as follows:

1. Clamp the action vector to unit length.
2. Scale it by `max_speed * dt`.
3. Apply the displacement to the current position.
4. Clamp the new position to the world bounds.
5. Recompute velocity from the position delta.

This means the action is interpreted as a desired 2D movement direction and magnitude, with the simulator enforcing the speed limit and boundaries.

## What Comes Out of `step`

At the Rust level, `World::step()` returns a `StepResult`:

- `events`: what happened during this step only
- `metrics`: cumulative metrics after the step

At the Python extension boundary, `SwarmWorld.step()` returns a dictionary with three top-level keys:

- `events`
- `metrics`
- `state`

### `events`

The per-step event payload currently includes:

- `targets_discovered`
- `agent_collisions`
- `obstacle_violations`
- `new_coverage_cells`

These are step-local counts and reset after each step. They are useful for reward calculation and for understanding what changed on this timestep.  

Their main goal is to drive reward terms tied to what happened on this step.

### `metrics`

The metrics payload includes cumulative episode-level values such as:

- `timestep`
- `target_count`
- `discovered_target_count`
- `target_discovery_rate`
- `coverage_ratio`
- `collision_count`
- `obstacle_violation_count`
- `connectivity_ratio`
- `average_communication_neighbors`
- `largest_connected_component_size`
- `communication_edge_count`
- `done`
- `all_targets_discovered`
- `horizon_reached`

These values are the main contract consumed by Python for logging, reporting, episode termination, and reward shaping.  

Their main goal is to drive episode-level logging, reporting, and termination logic

### `state`

The state snapshot gives Python the entity-level world state after the step (targets, obstacles, agents). That snapshot is used by the observation builder, evaluator, and renderer.  

Their main goal is to drive local observation construction and visualization.
