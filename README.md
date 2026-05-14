# DroneWatch

DroneWatch is a MARL engineering showcase where 16 cooperative drones learn, via RLlib PPO and later MAPPO-style centralized critic training, to discover targets in a partially observable continuous 2D environment with obstacles, collisions, local sensing, and short-range communication.

The repository is currently implementing Phase 2 of the project plan: a Python multi-agent environment wrapper around the completed Rust simulation core.

## Current Phase

Phase 2 establishes the RLlib/Gymnasium-facing environment layer before PPO training, OmegaConf configuration, and MLflow logging are added.

Implemented in this slice:

- Python package: `dronewatch`
- Rust extension module: `swarm_sim`
- Python wrapper: `dronewatch.sim`
- Rust world reset and step loop
- 16 default agents with continuous 2D movement
- bounded 100x100 world and fixed 200-step horizon
- static targets and circular no-fly obstacles
- target discovery, collision counts, obstacle violation counts
- coverage grid tracking
- communication connectivity metrics
- deterministic scripted rollout
- RLlib `MultiAgentEnv` wrapper: `SwarmSearchEnv`
- fixed-size local observations for all agents
- cooperative team reward calculation
- random policy baseline
- random rollout JSON report and GIF rendering
- uv dependency management
- PyO3/maturin build path
- Rust and Python tests
- Makefile and Docker skeleton

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Rust toolchain with Cargo
- Docker, optional for container validation

## Local Setup

Install Python dependencies and build the Rust extension into the uv environment:

```bash
make install
```

This runs:

```bash
uv sync --dev
uv run maturin develop -m rust/swarm_sim/Cargo.toml
```

## Smoke Check

Verify that Python can import the package, call into Rust, run a scripted simulator rollout, and run the random policy baseline:

```bash
uv run python -c "import dronewatch; print(dronewatch.__version__)"
uv run python -c "from dronewatch.sim import rust_version; print(rust_version())"
make rollout-rust
make rollout-random
make render-random
```

`make rollout-random` writes a random policy report to `artifacts/reports/random_policy_report.json`.
`make render-random` writes the same report plus `artifacts/gifs/random_policy_episode.gif`.

## Tests

Run the full Phase 2 test suite:

```bash
make test
```

Or run each side directly:

```bash
make test-rust
make test-python
```

## Docker

Build the Phase 0 validation image:

```bash
make docker-build
```

The image installs the uv-managed project, builds the PyO3 extension, and defaults to `make test`.
