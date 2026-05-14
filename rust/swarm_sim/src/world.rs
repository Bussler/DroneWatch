//! Core world state and step orchestration for the simulator.

use crate::{
    agent::Agent,
    config::{SimResult, SimulationConfig},
    coverage::CoverageGrid,
    geometry::Vec2,
    metrics::{CommunicationMetrics, EventCollector, SimulationMetrics, StepEvents},
    obstacle::Obstacle,
    scenario::ScenarioSampler,
    target::Target,
};

/// Result returned by a successful world step.
#[derive(Clone, Debug)]
pub struct StepResult {
    /// Events that happened during only this step.
    pub events: StepEvents,
    /// Metrics after this step has been applied.
    pub metrics: SimulationMetrics,
}

/// Mutable world state for a bounded 2D swarm-search episode.
///
/// `World` owns the current entities and episode counters. It orchestrates reset and step order,
/// while scenario generation and metric/event calculations live in dedicated modules.
#[derive(Clone, Debug)]
pub struct World {
    /// Immutable simulation settings used by this world instance.
    pub config: SimulationConfig,
    /// Agent states for the current scenario.
    pub agents: Vec<Agent>,
    /// Target states for the current scenario.
    pub targets: Vec<Target>,
    /// Circular no-fly obstacles for the current scenario.
    pub obstacles: Vec<Obstacle>,
    coverage: CoverageGrid,
    timestep: usize,
    collision_count: usize,
    obstacle_violation_count: usize,
    communication_metrics: CommunicationMetrics,
    done: bool,
}

impl World {
    /// Creates a world from a validated config and immediately resets it with `seed`.
    pub fn new(config: SimulationConfig, seed: u64) -> SimResult<Self> {
        config.validate()?;
        let coverage = CoverageGrid::new(config.coverage.grid_width, config.coverage.grid_height);
        let mut world = Self {
            config,
            agents: Vec::new(),
            targets: Vec::new(),
            obstacles: Vec::new(),
            coverage,
            timestep: 0,
            collision_count: 0,
            obstacle_violation_count: 0,
            communication_metrics: CommunicationMetrics::default(),
            done: false,
        };
        world.reset(seed)?;
        Ok(world)
    }

    /// Resets the episode state and generates a deterministic scenario from `seed`.
    ///
    /// Coverage is cleared and then seeded from the starting agent positions before metrics are
    /// returned. That initial coverage is not reported as a per-step event.
    pub fn reset(&mut self, seed: u64) -> SimResult<SimulationMetrics> {
        self.timestep = 0;
        self.collision_count = 0;
        self.obstacle_violation_count = 0;
        self.done = false;

        let scenario = ScenarioSampler::generate(&self.config, seed)?;
        self.agents = scenario.agents;
        self.targets = scenario.targets;
        self.obstacles = scenario.obstacles;
        self.coverage.reset();
        self.coverage.mark_from_agents(&self.agents, &self.config);
        self.communication_metrics = EventCollector::communication_graph(
            &self.agents,
            self.config.agents.communication_radius,
        );

        Ok(self.metrics())
    }

    /// Applies one action per agent and returns step events plus updated metrics.
    ///
    /// Each action is clamped to unit length, scaled by `max_speed * dt`, and the resulting position
    /// is clamped to the configured world bounds. Calling `step` after the episode is done returns
    /// an error rather than a no-op result.
    pub fn step(&mut self, actions: &[Vec2]) -> SimResult<StepResult> {
        if self.done {
            return Err(
                "cannot step a world after the episode is done; call reset first".to_string(),
            );
        }
        if actions.len() != self.agents.len() {
            return Err(format!(
                "expected {} actions, received {}",
                self.agents.len(),
                actions.len()
            ));
        }
        if let Some((index, _)) = actions
            .iter()
            .enumerate()
            .find(|(_, action)| !action.is_finite())
        {
            return Err(format!("action {index} contains a non-finite value"));
        }

        self.apply_actions(actions);

        let mut events: StepEvents = StepEvents::default();
        events.targets_discovered = self.update_targets();
        events.agent_collisions = EventCollector::count_agent_collisions(
            &self.agents,
            self.config.agents.collision_radius,
        );
        events.obstacle_violations = EventCollector::count_obstacle_violations(
            &self.agents,
            &self.obstacles,
            self.config.agents.collision_radius,
        );
        events.new_coverage_cells = self.coverage.mark_from_agents(&self.agents, &self.config);

        self.collision_count += events.agent_collisions;
        self.obstacle_violation_count += events.obstacle_violations;
        self.communication_metrics = EventCollector::communication_graph(
            &self.agents,
            self.config.agents.communication_radius,
        );
        self.timestep += 1;

        self.done = EventCollector::all_targets_discovered(&self.targets)
            || self.timestep >= self.config.max_episode_steps;

        Ok(StepResult {
            events,
            metrics: self.metrics(),
        })
    }

    /// Returns current cumulative metrics without mutating world state.
    pub fn metrics(&self) -> SimulationMetrics {
        let discovered_target_count = EventCollector::discovered_target_count(&self.targets);
        let target_count = self.targets.len();
        let all_targets_discovered = EventCollector::all_targets_discovered(&self.targets);
        let horizon_reached = self.timestep >= self.config.max_episode_steps;
        SimulationMetrics {
            timestep: self.timestep,
            max_episode_steps: self.config.max_episode_steps,
            target_count,
            discovered_target_count,
            target_discovery_rate: if target_count == 0 {
                1.0
            } else {
                discovered_target_count as f64 / target_count as f64
            },
            coverage_ratio: self.coverage.ratio(),
            covered_cells: self.coverage.covered_cells(),
            total_coverage_cells: self.coverage.total_cells(),
            collision_count: self.collision_count,
            obstacle_violation_count: self.obstacle_violation_count,
            connectivity_ratio: self.communication_metrics.connectivity_ratio,
            average_communication_neighbors: self.communication_metrics.average_neighbors,
            largest_connected_component_size: self
                .communication_metrics
                .largest_connected_component_size,
            communication_edge_count: self.communication_metrics.edge_count,
            done: self.done,
            all_targets_discovered,
            horizon_reached,
        }
    }

    /// Returns the current episode timestep without recomputing metrics.
    pub fn timestep(&self) -> usize {
        self.timestep
    }

    /// Returns whether this episode has reached a terminal or truncated state.
    pub fn is_done(&self) -> bool {
        self.done
    }

    fn apply_actions(&mut self, actions: &[Vec2]) {
        let max_displacement = self.config.agents.max_speed * self.config.world.dt;
        for (agent, action) in self.agents.iter_mut().zip(actions.iter()) {
            let previous_position = agent.position;
            let displacement = action.clamp_length(1.0) * max_displacement;
            agent.position = (agent.position + displacement)
                .clamp_to_bounds(self.config.world.width, self.config.world.height);
            agent.velocity = (agent.position - previous_position) * (1.0 / self.config.world.dt);
        }
    }

    fn update_targets(&mut self) -> usize {
        let mut newly_discovered = 0;
        for target in &mut self.targets {
            if target.discovered {
                continue;
            }

            if self.agents.iter().any(|agent| {
                agent.position.distance(target.position) <= self.config.targets.discovery_radius
            }) {
                target.discovered = true;
                newly_discovered += 1;
            }
        }
        newly_discovered
    }
}

#[cfg(test)]
mod tests {
    use crate::{agent::Agent, config::SimulationConfig, geometry::Vec2, target::Target};

    use super::World;

    #[test]
    fn reset_initializes_default_world() {
        let world = World::new(SimulationConfig::default(), 7).unwrap();

        assert_eq!(world.agents.len(), 16);
        assert_eq!(world.targets.len(), 20);
        assert_eq!(world.obstacles.len(), 8);
        assert_eq!(world.metrics().timestep, 0);
        assert_eq!(world.timestep(), 0);
    }

    #[test]
    fn step_moves_agents_and_increments_timestep() {
        let mut world = World::new(SimulationConfig::default(), 7).unwrap();
        let previous_position = world.agents[0].position;
        let actions = vec![Vec2::new(1.0, 0.0); world.agents.len()];

        let result = world.step(&actions).unwrap();

        assert_eq!(result.metrics.timestep, 1);
        assert_eq!(world.timestep(), 1);
        assert!(world.agents[0].position.x >= previous_position.x);
        assert!(world.agents.iter().all(|agent| agent.position.is_finite()));
    }

    #[test]
    fn rejects_wrong_action_count() {
        let mut world = World::new(SimulationConfig::default(), 7).unwrap();
        let error = world.step(&[Vec2::new(0.0, 0.0)]).unwrap_err();

        assert!(error.contains("expected 16 actions"));
    }

    #[test]
    fn step_discovers_nearby_targets_once() {
        let mut world = World::new(SimulationConfig::default(), 7).unwrap();
        world.agents = vec![Agent::new(0, Vec2::new(10.0, 10.0))];
        world.targets = vec![
            Target::new(0, Vec2::new(11.0, 10.0)),
            Target::new(1, Vec2::new(50.0, 50.0)),
        ];
        world.obstacles = Vec::new();

        let first_step = world.step(&[Vec2::zero()]).unwrap();

        assert_eq!(first_step.events.targets_discovered, 1);
        assert_eq!(first_step.metrics.discovered_target_count, 1);
        assert!(!first_step.metrics.done);
        assert!(world.targets[0].discovered);
        assert!(!world.targets[1].discovered);

        let second_step = world.step(&[Vec2::zero()]).unwrap();

        assert_eq!(second_step.events.targets_discovered, 0);
        assert_eq!(second_step.metrics.discovered_target_count, 1);
        assert!(world.targets[0].discovered);
        assert!(!world.targets[1].discovered);
    }

    #[test]
    fn full_scripted_rollout_reaches_horizon() {
        let mut world = World::new(SimulationConfig::default(), 11).unwrap();
        let mut last_metrics = world.metrics();

        while !world.is_done() {
            let actions: Vec<Vec2> = (0..world.agents.len())
                .map(|index| {
                    if index % 2 == 0 {
                        Vec2::new(1.0, 0.25)
                    } else {
                        Vec2::new(-0.25, 1.0)
                    }
                })
                .collect();
            last_metrics = world.step(&actions).unwrap().metrics;
        }

        assert!(last_metrics.timestep <= 200);
        assert!((0.0..=1.0).contains(&last_metrics.coverage_ratio));
        assert!((0.0..=1.0).contains(&last_metrics.connectivity_ratio));
        assert!(world.step(&vec![Vec2::zero(); world.agents.len()]).is_err());
    }
}
