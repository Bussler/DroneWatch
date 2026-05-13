use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    agent::Agent,
    config::SimulationConfig,
    coverage::CoverageGrid,
    geometry::{circles_overlap, Vec2},
    metrics::{CommunicationMetrics, SimulationMetrics, StepEvents},
    obstacle::Obstacle,
    target::Target,
};

pub type SimResult<T> = Result<T, String>;

#[derive(Clone, Debug)]
pub struct StepResult {
    pub events: StepEvents,
    pub metrics: SimulationMetrics,
}

#[derive(Clone, Debug)]
pub struct World {
    pub config: SimulationConfig,
    pub agents: Vec<Agent>,
    pub targets: Vec<Target>,
    pub obstacles: Vec<Obstacle>,
    coverage: CoverageGrid,
    timestep: usize,
    collision_count: usize,
    obstacle_violation_count: usize,
    communication_metrics: CommunicationMetrics,
    done: bool,
}

impl World {
    pub fn new(config: SimulationConfig, seed: u64) -> SimResult<Self> {
        validate_config(&config)?;
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

    pub fn reset(&mut self, seed: u64) -> SimResult<SimulationMetrics> {
        let mut rng = StdRng::seed_from_u64(seed);
        self.timestep = 0;
        self.collision_count = 0;
        self.obstacle_violation_count = 0;
        self.done = false;

        self.obstacles = self.sample_obstacles(&mut rng)?;
        self.agents = self.sample_agents(&mut rng)?;
        self.targets = self.sample_targets(&mut rng)?;
        self.coverage.reset();
        self.coverage.mark_from_agents(&self.agents, &self.config);
        self.communication_metrics = self.compute_communication_metrics();

        Ok(self.metrics())
    }

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

        let mut events = StepEvents::default();
        events.targets_discovered = self.update_targets();
        events.agent_collisions = self.count_agent_collisions();
        events.obstacle_violations = self.count_obstacle_violations();
        events.new_coverage_cells = self.coverage.mark_from_agents(&self.agents, &self.config);

        self.collision_count += events.agent_collisions;
        self.obstacle_violation_count += events.obstacle_violations;
        self.communication_metrics = self.compute_communication_metrics();
        self.timestep += 1;

        self.done = self.all_targets_discovered() || self.timestep >= self.config.max_episode_steps;

        Ok(StepResult {
            events,
            metrics: self.metrics(),
        })
    }

    pub fn metrics(&self) -> SimulationMetrics {
        let discovered_target_count = self.discovered_target_count();
        let target_count = self.targets.len();
        let all_targets_discovered = target_count > 0 && discovered_target_count == target_count;
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

    fn count_agent_collisions(&self) -> usize {
        let mut collisions = 0;
        for left_index in 0..self.agents.len() {
            for right_index in (left_index + 1)..self.agents.len() {
                if circles_overlap(
                    self.agents[left_index].position,
                    self.config.agents.collision_radius,
                    self.agents[right_index].position,
                    self.config.agents.collision_radius,
                ) {
                    collisions += 1;
                }
            }
        }
        collisions
    }

    fn count_obstacle_violations(&self) -> usize {
        self.agents
            .iter()
            .flat_map(|agent| {
                self.obstacles.iter().map(move |obstacle| {
                    circles_overlap(
                        agent.position,
                        self.config.agents.collision_radius,
                        obstacle.position,
                        obstacle.radius,
                    )
                })
            })
            .filter(|overlaps| *overlaps)
            .count()
    }

    fn compute_communication_metrics(&self) -> CommunicationMetrics {
        let agent_count = self.agents.len();
        if agent_count == 0 {
            return CommunicationMetrics::default();
        }

        let mut adjacency = vec![Vec::<usize>::new(); agent_count];
        let mut edge_count = 0;

        for left_index in 0..agent_count {
            for right_index in (left_index + 1)..agent_count {
                if self.agents[left_index]
                    .position
                    .distance(self.agents[right_index].position)
                    <= self.config.agents.communication_radius
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

    fn discovered_target_count(&self) -> usize {
        self.targets
            .iter()
            .filter(|target| target.discovered)
            .count()
    }

    fn all_targets_discovered(&self) -> bool {
        !self.targets.is_empty() && self.discovered_target_count() == self.targets.len()
    }

    fn sample_obstacles(&self, rng: &mut StdRng) -> SimResult<Vec<Obstacle>> {
        let mut obstacles = Vec::with_capacity(self.config.obstacles.count);
        for id in 0..self.config.obstacles.count {
            let radius =
                rng.gen_range(self.config.obstacles.min_radius..=self.config.obstacles.max_radius);
            let position = self.sample_bounded_point(rng, radius)?;
            obstacles.push(Obstacle::new(id, position, radius));
        }
        Ok(obstacles)
    }

    fn sample_agents(&self, rng: &mut StdRng) -> SimResult<Vec<Agent>> {
        let mut agents = Vec::with_capacity(self.config.agents.count);
        for id in 0..self.config.agents.count {
            let position = self.sample_point_away_from_obstacles(
                rng,
                self.config.agents.collision_radius,
                self.config.agents.collision_radius,
            )?;
            agents.push(Agent::new(id, position));
        }
        Ok(agents)
    }

    fn sample_targets(&self, rng: &mut StdRng) -> SimResult<Vec<Target>> {
        let mut targets = Vec::with_capacity(self.config.targets.count);
        for id in 0..self.config.targets.count {
            let position = self.sample_point_away_from_obstacles(
                rng,
                self.config.targets.discovery_radius,
                self.config.targets.discovery_radius,
            )?;
            targets.push(Target::new(id, position));
        }
        Ok(targets)
    }

    fn sample_bounded_point(&self, rng: &mut StdRng, margin: f64) -> SimResult<Vec2> {
        if self.config.world.width <= margin * 2.0 || self.config.world.height <= margin * 2.0 {
            return Err("world is too small for configured entity radius".to_string());
        }
        Ok(Vec2::new(
            rng.gen_range(margin..=(self.config.world.width - margin)),
            rng.gen_range(margin..=(self.config.world.height - margin)),
        ))
    }

    fn sample_point_away_from_obstacles(
        &self,
        rng: &mut StdRng,
        margin: f64,
        clearance: f64,
    ) -> SimResult<Vec2> {
        for _ in 0..10_000 {
            let position = self.sample_bounded_point(rng, margin)?;
            if self
                .obstacles
                .iter()
                .all(|obstacle| position.distance(obstacle.position) > obstacle.radius + clearance)
            {
                return Ok(position);
            }
        }

        Err("could not place entity away from obstacles after 10000 attempts".to_string())
    }
}

fn validate_config(config: &SimulationConfig) -> SimResult<()> {
    if config.max_episode_steps == 0 {
        return Err("max_episode_steps must be greater than zero".to_string());
    }
    if config.agents.count == 0 {
        return Err("agent count must be greater than zero".to_string());
    }
    if config.world.width <= 0.0 || config.world.height <= 0.0 || config.world.dt <= 0.0 {
        return Err("world width, height, and dt must be positive".to_string());
    }
    if config.agents.max_speed <= 0.0
        || config.agents.collision_radius <= 0.0
        || config.agents.communication_radius <= 0.0
        || config.agents.sensing_radius <= 0.0
    {
        return Err("agent radii and max speed must be positive".to_string());
    }
    if config.targets.discovery_radius <= 0.0 {
        return Err("target discovery radius must be positive".to_string());
    }
    if config.obstacles.min_radius <= 0.0
        || config.obstacles.max_radius < config.obstacles.min_radius
    {
        return Err("obstacle radii must be positive and ordered".to_string());
    }
    if config.coverage.grid_width == 0
        || config.coverage.grid_height == 0
        || config.coverage.sensing_radius <= 0.0
    {
        return Err("coverage grid dimensions and sensing radius must be positive".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{config::SimulationConfig, geometry::Vec2};

    use super::World;

    #[test]
    fn reset_initializes_default_world() {
        let world = World::new(SimulationConfig::default(), 7).unwrap();

        assert_eq!(world.agents.len(), 16);
        assert_eq!(world.targets.len(), 20);
        assert_eq!(world.obstacles.len(), 8);
        assert_eq!(world.metrics().timestep, 0);
    }

    #[test]
    fn step_moves_agents_and_increments_timestep() {
        let mut world = World::new(SimulationConfig::default(), 7).unwrap();
        let previous_position = world.agents[0].position;
        let actions = vec![Vec2::new(1.0, 0.0); world.agents.len()];

        let result = world.step(&actions).unwrap();

        assert_eq!(result.metrics.timestep, 1);
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
