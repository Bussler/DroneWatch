//! Event and metric data structures plus pure metric calculations.

use crate::{agent::Agent, geometry::circles_overlap, obstacle::Obstacle, target::Target};

/// Per-step events produced by one successful world step.
#[derive(Clone, Debug, Default)]
pub struct StepEvents {
    /// Number of newly discovered targets during this step.
    pub targets_discovered: usize,
    /// Number of unordered agent-agent collision pairs during this step.
    pub agent_collisions: usize,
    /// Number of agent-obstacle overlaps during this step.
    pub obstacle_violations: usize,
    /// Number of coverage cells newly marked by this step.
    pub new_coverage_cells: usize,
}

/// Communication graph metrics for the current agent positions.
#[derive(Clone, Debug)]
pub struct CommunicationMetrics {
    /// Number of agents in the largest connected communication component.
    pub largest_connected_component_size: usize,
    /// Largest connected component size divided by the number of agents.
    pub connectivity_ratio: f64,
    /// Mean number of communication neighbors per agent.
    pub average_neighbors: f64,
    /// Number of undirected communication edges.
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

/// Cumulative and current simulator metrics exposed to Python and tests.
#[derive(Clone, Debug)]
pub struct SimulationMetrics {
    /// Current timestep after successful steps.
    pub timestep: usize,
    /// Maximum episode horizon configured for the world.
    pub max_episode_steps: usize,
    /// Number of targets in the current scenario.
    pub target_count: usize,
    /// Number of targets discovered so far.
    pub discovered_target_count: usize,
    /// Fraction of targets discovered so far.
    pub target_discovery_rate: f64,
    /// Fraction of coverage grid cells marked covered so far.
    pub coverage_ratio: f64,
    /// Number of covered grid cells.
    pub covered_cells: usize,
    /// Total number of coverage grid cells.
    pub total_coverage_cells: usize,
    /// Cumulative count of agent-agent collision pairs.
    pub collision_count: usize,
    /// Cumulative count of agent-obstacle overlap violations.
    pub obstacle_violation_count: usize,
    /// Current communication connectivity ratio.
    pub connectivity_ratio: f64,
    /// Current mean communication neighbors per agent.
    pub average_communication_neighbors: f64,
    /// Current largest connected component size.
    pub largest_connected_component_size: usize,
    /// Current undirected communication edge count.
    pub communication_edge_count: usize,
    /// Whether the episode has ended.
    pub done: bool,
    /// Whether all targets have been discovered.
    pub all_targets_discovered: bool,
    /// Whether the fixed episode horizon has been reached.
    pub horizon_reached: bool,
}

/// Stateless helper for event detection and graph metrics over world state.
pub struct EventCollector;

impl EventCollector {
    /// Counts unordered agent-agent collision pairs.
    pub fn count_agent_collisions(agents: &[Agent], collision_radius: f64) -> usize {
        let mut collisions = 0;
        for left_index in 0..agents.len() {
            for right_index in (left_index + 1)..agents.len() {
                if circles_overlap(
                    agents[left_index].position,
                    collision_radius,
                    agents[right_index].position,
                    collision_radius,
                ) {
                    collisions += 1;
                }
            }
        }
        collisions
    }

    /// Counts agent-obstacle overlaps for circular no-fly zones.
    pub fn count_obstacle_violations(
        agents: &[Agent],
        obstacles: &[Obstacle],
        agent_radius: f64,
    ) -> usize {
        agents
            .iter()
            .flat_map(|agent| {
                obstacles.iter().map(move |obstacle| {
                    circles_overlap(
                        agent.position,
                        agent_radius,
                        obstacle.position,
                        obstacle.radius,
                    )
                })
            })
            .filter(|overlaps| *overlaps)
            .count()
    }

    /// Computes communication graph metrics using the configured communication radius.
    pub fn communication_graph(
        agents: &[Agent],
        communication_radius: f64,
    ) -> CommunicationMetrics {
        let agent_count = agents.len();
        if agent_count == 0 {
            return CommunicationMetrics::default();
        }

        let mut adjacency = vec![Vec::<usize>::new(); agent_count];
        let mut edge_count = 0;

        for left_index in 0..agent_count {
            for right_index in (left_index + 1)..agent_count {
                if agents[left_index]
                    .position
                    .distance(agents[right_index].position)
                    <= communication_radius
                {
                    adjacency[left_index].push(right_index);
                    adjacency[right_index].push(left_index);
                    edge_count += 1;
                }
            }
        }

        let mut visited = vec![false; agent_count];
        let mut largest_component = 0;
        for start in 0..agent_count {
            if visited[start] {
                continue;
            }

            let mut stack = vec![start];
            visited[start] = true;
            let mut component_size = 0;

            while let Some(node) = stack.pop() {
                component_size += 1;
                for neighbor in &adjacency[node] {
                    if !visited[*neighbor] {
                        visited[*neighbor] = true;
                        stack.push(*neighbor);
                    }
                }
            }

            largest_component = largest_component.max(component_size);
        }

        CommunicationMetrics {
            largest_connected_component_size: largest_component,
            connectivity_ratio: largest_component as f64 / agent_count as f64,
            average_neighbors: (edge_count * 2) as f64 / agent_count as f64,
            edge_count,
        }
    }

    /// Counts targets that have already been discovered.
    pub fn discovered_target_count(targets: &[Target]) -> usize {
        targets.iter().filter(|target| target.discovered).count()
    }

    /// Returns true when every target in a target set is discovered.
    pub fn all_targets_discovered(targets: &[Target]) -> bool {
        Self::discovered_target_count(targets) == targets.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::{agent::Agent, geometry::Vec2, obstacle::Obstacle};

    use super::EventCollector;

    #[test]
    fn collision_count_uses_unordered_pairs() {
        let agents = vec![
            Agent::new(0, Vec2::new(0.0, 0.0)),
            Agent::new(1, Vec2::new(1.0, 0.0)),
            Agent::new(2, Vec2::new(5.0, 0.0)),
        ];

        assert_eq!(EventCollector::count_agent_collisions(&agents, 0.75), 1);
    }

    #[test]
    fn obstacle_violations_count_agent_obstacle_overlaps() {
        let agents = vec![Agent::new(0, Vec2::new(0.0, 0.0))];
        let obstacles = vec![Obstacle::new(0, Vec2::new(1.0, 0.0), 1.0)];

        assert_eq!(
            EventCollector::count_obstacle_violations(&agents, &obstacles, 0.5),
            1
        );
    }

    #[test]
    fn communication_graph_tracks_largest_component() {
        let agents = vec![
            Agent::new(0, Vec2::new(0.0, 0.0)),
            Agent::new(1, Vec2::new(1.0, 0.0)),
            Agent::new(2, Vec2::new(20.0, 0.0)),
        ];

        let metrics = EventCollector::communication_graph(&agents, 2.0);

        assert_eq!(metrics.edge_count, 1);
        assert_eq!(metrics.largest_connected_component_size, 2);
        assert!((metrics.connectivity_ratio - (2.0 / 3.0)).abs() < 1e-9);
    }
}
