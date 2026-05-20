//! PyO3 bindings and public Rust modules for the DroneWatch simulation core.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::random;

pub mod agent;
pub mod config;
pub mod coverage;
pub mod geometry;
pub mod metrics;
pub mod obstacle;
pub mod scenario;
pub mod target;
pub mod world;

use crate::{
    config::SimulationConfig,
    geometry::Vec2,
    metrics::{SimulationMetrics, StepEvents},
    world::World,
};

/// Returns the Rust crate version compiled into the extension.
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn resolve_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(random)
}

#[pyfunction]
fn version() -> &'static str {
    crate_version()
}

#[pyclass]
/// Python-owned wrapper around the Rust `World` simulation state.
struct SwarmWorld {
    world: World,
}

#[pymethods]
impl SwarmWorld {
    #[new]
    #[pyo3(signature = (seed=None, config=None))]
    fn new(seed: Option<u64>, config: Option<String>) -> PyResult<Self> {
        let simulation_config = match config {
            Some(json) => simulation_config_from_json(&json)?,
            None => SimulationConfig::default(),
        };
        let world = World::new(simulation_config, resolve_seed(seed))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { world })
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = self
            .world
            .reset(resolve_seed(seed))
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        metrics_to_py(py, &result)
    }

    fn step(&mut self, actions: Vec<(f64, f64)>, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let action_vectors: Vec<Vec2> = actions.into_iter().map(|(x, y)| Vec2::new(x, y)).collect();
        let result = self
            .world
            .step(&action_vectors)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let dict = PyDict::new_bound(py);
        dict.set_item("events", step_events_to_py_bound(py, &result.events)?)?;
        dict.set_item("metrics", metrics_to_py_bound(py, &result.metrics)?)?;
        dict.set_item("state", self.state(py)?)?;
        Ok(dict.into())
    }

    fn state(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("timestep", self.world.timestep())?;

        let agents = PyList::empty_bound(py);
        for agent in &self.world.agents {
            let item = PyDict::new_bound(py);
            item.set_item("id", agent.id)?;
            item.set_item("position", [agent.position.x, agent.position.y])?;
            item.set_item("velocity", [agent.velocity.x, agent.velocity.y])?;
            agents.append(item)?;
        }
        dict.set_item("agents", agents)?;

        let targets = PyList::empty_bound(py);
        for target in &self.world.targets {
            let item = PyDict::new_bound(py);
            item.set_item("id", target.id)?;
            item.set_item("position", [target.position.x, target.position.y])?;
            item.set_item("discovered", target.discovered)?;
            targets.append(item)?;
        }
        dict.set_item("targets", targets)?;

        let obstacles = PyList::empty_bound(py);
        for obstacle in &self.world.obstacles {
            let item = PyDict::new_bound(py);
            item.set_item("id", obstacle.id)?;
            item.set_item("position", [obstacle.position.x, obstacle.position.y])?;
            item.set_item("radius", obstacle.radius)?;
            obstacles.append(item)?;
        }
        dict.set_item("obstacles", obstacles)?;

        Ok(dict.into())
    }

    fn metrics(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        metrics_to_py(py, &self.world.metrics())
    }

    fn is_done(&self) -> bool {
        self.world.is_done()
    }
}

fn simulation_config_from_json(config_json: &str) -> PyResult<SimulationConfig> {
    let config: SimulationConfig = serde_json::from_str(config_json).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid simulation config JSON: {error}"))
    })?;
    config.validate().map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid simulation config: {error}"))
    })?;
    Ok(config)
}

fn metrics_to_py(py: Python<'_>, metrics: &SimulationMetrics) -> PyResult<Py<PyDict>> {
    Ok(metrics_to_py_bound(py, metrics)?.into())
}

fn step_events_to_py_bound<'py>(
    py: Python<'py>,
    events: &StepEvents,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("targets_discovered", events.targets_discovered)?;
    dict.set_item("agent_collisions", events.agent_collisions)?;
    dict.set_item("obstacle_violations", events.obstacle_violations)?;
    dict.set_item("new_coverage_cells", events.new_coverage_cells)?;
    Ok(dict)
}

fn metrics_to_py_bound<'py>(
    py: Python<'py>,
    metrics: &SimulationMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("timestep", metrics.timestep)?;
    dict.set_item("max_episode_steps", metrics.max_episode_steps)?;
    dict.set_item("target_count", metrics.target_count)?;
    dict.set_item("discovered_target_count", metrics.discovered_target_count)?;
    dict.set_item("target_discovery_rate", metrics.target_discovery_rate)?;
    dict.set_item("coverage_ratio", metrics.coverage_ratio)?;
    dict.set_item("covered_cells", metrics.covered_cells)?;
    dict.set_item("total_coverage_cells", metrics.total_coverage_cells)?;
    dict.set_item("collision_count", metrics.collision_count)?;
    dict.set_item("obstacle_violation_count", metrics.obstacle_violation_count)?;
    dict.set_item("connectivity_ratio", metrics.connectivity_ratio)?;
    dict.set_item(
        "average_communication_neighbors",
        metrics.average_communication_neighbors,
    )?;
    dict.set_item(
        "largest_connected_component_size",
        metrics.largest_connected_component_size,
    )?;
    dict.set_item("communication_edge_count", metrics.communication_edge_count)?;
    dict.set_item("done", metrics.done)?;
    dict.set_item("all_targets_discovered", metrics.all_targets_discovered)?;
    dict.set_item("horizon_reached", metrics.horizon_reached)?;
    Ok(dict)
}

#[pymodule]
fn swarm_sim(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_class::<SwarmWorld>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn crate_version_matches_package_version() {
        assert_eq!(crate_version(), "0.1.0");
    }
}
