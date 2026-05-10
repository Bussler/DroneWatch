# DroneWatch

DroneWatch is a MARL engineering showcase where 16 cooperative drones learn, via RLlib PPO and later MAPPO-style centralized critic training, to discover targets in a partially observable continuous 2D environment with obstacles, collisions, local sensing, and short-range communication.

The repository is currently implementing Phase 0 of the project plan: a uv-managed Python package, a Rust PyO3 extension crate, maturin-based local builds, and smoke tests that prove Python can call Rust.

## Current Phase

Phase 0 establishes the project skeleton before simulator or training logic is added.

Implemented in this slice:

- Python package: `dronewatch`
- Rust extension module: `swarm_sim`
- Python wrapper: `dronewatch.sim`
- uv dependency management
- PyO3/maturin build path
- Rust and Python smoke tests
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

Verify that Python can import the package and call into Rust:

```bash
uv run python -c "import dronewatch; print(dronewatch.__version__)"
uv run python -c "from dronewatch.sim import rust_version; print(rust_version())"
```

## Tests

Run the full Phase 0 test suite:

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
