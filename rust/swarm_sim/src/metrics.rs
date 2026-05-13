#[derive(Clone, Debug, Default)]
pub struct StepEvents {
    pub targets_discovered: usize,
    pub agent_collisions: usize,
    pub obstacle_violations: usize,
    pub new_coverage_cells: usize,
}

#[derive(Clone, Debug)]
pub struct CommunicationMetrics {
    pub largest_connected_component_size: usize,
    pub connectivity_ratio: f64,
    pub average_neighbors: f64,
    pub edge_count: usize,
}

impl Default for CommunicationMetrics {
    fn default() -> Self {
        Self {
            largest_connected_component_size: 0,
            connectivity_ratio: 0.0,
            average_neighbors: 0.0,
            edge_count: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimulationMetrics {
    pub timestep: usize,
    pub max_episode_steps: usize,
    pub target_count: usize,
    pub discovered_target_count: usize,
    pub target_discovery_rate: f64,
    pub coverage_ratio: f64,
    pub covered_cells: usize,
    pub total_coverage_cells: usize,
    pub collision_count: usize,
    pub obstacle_violation_count: usize,
    pub connectivity_ratio: f64,
    pub average_communication_neighbors: f64,
    pub largest_connected_component_size: usize,
    pub communication_edge_count: usize,
    pub done: bool,
    pub all_targets_discovered: bool,
    pub horizon_reached: bool,
}
