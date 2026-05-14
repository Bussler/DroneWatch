# DroneWatch

DroneWatch is a MARL engineering showcase where 16 cooperative drones learn, via RLlib PPO and later MAPPO-style centralized critic training, to discover targets in a partially observable continuous 2D environment with obstacles, collisions, local sensing, and short-range communication.

The repository is currently implementing Phase 1 of the project plan: a Rust simulation core with Python access through PyO3, building on the Phase 0 package skeleton.

## Current Phase

Phase 1 establishes the first working simulator slice before RLlib environment, reward, rendering, and training logic are added.

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

Verify that Python can import the package, call into Rust, and run a scripted simulator rollout:

```bash
uv run python -c "import dronewatch; print(dronewatch.__version__)"
uv run python -c "from dronewatch.sim import rust_version; print(rust_version())"
make rollout-rust
```

## Tests

Run the full Phase 1 test suite:

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
