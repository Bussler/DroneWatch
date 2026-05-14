//! Seeded scenario generation for agents, targets, and obstacles.

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    agent::Agent,
    config::{SimResult, SimulationConfig},
    geometry::Vec2,
    obstacle::Obstacle,
    target::Target,
};

/// Entities produced by a reset-time scenario generation pass.
#[derive(Clone, Debug)]
pub struct GeneratedScenario {
    /// Agents placed in the world.
    pub agents: Vec<Agent>,
    /// Static targets placed in the world.
    pub targets: Vec<Target>,
    /// Circular no-fly obstacles placed in the world.
    pub obstacles: Vec<Obstacle>,
}

/// Generates deterministic scenarios from a simulation config and seed.
pub struct ScenarioSampler;

impl ScenarioSampler {
    /// Generates a complete scenario using bounded random placement.
    pub fn generate(config: &SimulationConfig, seed: u64) -> SimResult<GeneratedScenario> {
        let mut rng = StdRng::seed_from_u64(seed);
        let obstacles = sample_obstacles(config, &mut rng)?;
        let agents = sample_agents(config, &obstacles, &mut rng)?;
        let targets = sample_targets(config, &obstacles, &mut rng)?;

        Ok(GeneratedScenario {
            agents,
            targets,
            obstacles,
        })
    }
}

fn sample_obstacles(config: &SimulationConfig, rng: &mut StdRng) -> SimResult<Vec<Obstacle>> {
    let mut obstacles = Vec::with_capacity(config.obstacles.count);
    for id in 0..config.obstacles.count {
        let radius = rng.gen_range(config.obstacles.min_radius..=config.obstacles.max_radius);
        let position = sample_bounded_point(config, rng, radius)?;
        obstacles.push(Obstacle::new(id, position, radius));
    }
    Ok(obstacles)
}

fn sample_agents(
    config: &SimulationConfig,
    obstacles: &[Obstacle],
    rng: &mut StdRng,
) -> SimResult<Vec<Agent>> {
    let mut agents: Vec<Agent> = Vec::with_capacity(config.agents.count);
    let mut reserved_positions: Vec<Vec2> = Vec::with_capacity(config.agents.count);
    for id in 0..config.agents.count {
        let position = sample_point_with_clearance(
            config,
            obstacles,
            &reserved_positions,
            rng,
            config.agents.collision_radius,
            config.agents.collision_radius,
            config.agents.collision_radius * 2.0,
        )?;
        reserved_positions.push(position);
        agents.push(Agent::new(id, position));
    }
    Ok(agents)
}

fn sample_targets(
    config: &SimulationConfig,
    obstacles: &[Obstacle],
    rng: &mut StdRng,
) -> SimResult<Vec<Target>> {
    let mut targets: Vec<Target> = Vec::with_capacity(config.targets.count);
    let mut reserved_positions: Vec<Vec2> = Vec::with_capacity(config.targets.count);
    for id in 0..config.targets.count {
        let position = sample_point_with_clearance(
            config,
            obstacles,
            &reserved_positions,
            rng,
            config.targets.discovery_radius,
            config.targets.discovery_radius,
            config.targets.discovery_radius,
        )?;
        reserved_positions.push(position);
        targets.push(Target::new(id, position));
    }
    Ok(targets)
}

fn sample_bounded_point(
    config: &SimulationConfig,
    rng: &mut StdRng,
    margin: f64,
) -> SimResult<Vec2> {
    if config.world.width <= margin * 2.0 || config.world.height <= margin * 2.0 {
        return Err("world is too small for configured entity radius".to_string());
    }
    Ok(Vec2::new(
        rng.gen_range(margin..=(config.world.width - margin)),
        rng.gen_range(margin..=(config.world.height - margin)),
    ))
}

fn sample_point_with_clearance(
    config: &SimulationConfig,
    obstacles: &[Obstacle],
    reserved_positions: &[Vec2],
    rng: &mut StdRng,
    margin: f64,
    obstacle_clearance: f64,
    point_clearance: f64,
) -> SimResult<Vec2> {
    for _ in 0..10_000 {
        let position = sample_bounded_point(config, rng, margin)?;
        let clear_of_obstacles = obstacles.iter().all(|obstacle| {
            position.distance(obstacle.position) > obstacle.radius + obstacle_clearance
        });
        let clear_of_reserved_points = reserved_positions
            .iter()
            .all(|reserved_position| position.distance(*reserved_position) > point_clearance);

        if clear_of_obstacles && clear_of_reserved_points {
            return Ok(position);
        }
    }

    Err("could not place entity with required clearance after 10000 attempts".to_string())
}

#[cfg(test)]
mod tests {
    use crate::config::SimulationConfig;

    use super::ScenarioSampler;

    #[test]
    fn generated_scenario_matches_config_counts() {
        let config = SimulationConfig::default();
        let scenario = ScenarioSampler::generate(&config, 7).unwrap();

        assert_eq!(scenario.agents.len(), config.agents.count);
        assert_eq!(scenario.targets.len(), config.targets.count);
        assert_eq!(scenario.obstacles.len(), config.obstacles.count);
    }

    #[test]
    fn generation_is_deterministic_for_same_seed() {
        let config = SimulationConfig::default();
        let first = ScenarioSampler::generate(&config, 7).unwrap();
        let second = ScenarioSampler::generate(&config, 7).unwrap();

        assert_eq!(first.agents[0].position, second.agents[0].position);
        assert_eq!(first.targets[0].position, second.targets[0].position);
        assert_eq!(first.obstacles[0].position, second.obstacles[0].position);
    }

    #[test]
    fn generated_positions_are_inside_world_bounds() {
        let config = SimulationConfig::default();
        let scenario = ScenarioSampler::generate(&config, 7).unwrap();

        assert!(scenario.agents.iter().all(|agent| {
            (0.0..=config.world.width).contains(&agent.position.x)
                && (0.0..=config.world.height).contains(&agent.position.y)
        }));
        assert!(scenario.targets.iter().all(|target| {
            (0.0..=config.world.width).contains(&target.position.x)
                && (0.0..=config.world.height).contains(&target.position.y)
        }));
    }

    #[test]
    fn generated_agents_do_not_spawn_overlapping() {
        let config = SimulationConfig::default();
        let scenario = ScenarioSampler::generate(&config, 7).unwrap();
        let minimum_distance = config.agents.collision_radius * 2.0;

        for left_index in 0..scenario.agents.len() {
            for right_index in (left_index + 1)..scenario.agents.len() {
                let distance = scenario.agents[left_index]
                    .position
                    .distance(scenario.agents[right_index].position);
                assert!(distance > minimum_distance);
            }
        }
    }

    #[test]
    fn generated_targets_do_not_spawn_inside_other_target_discovery_radii() {
        let config = SimulationConfig::default();
        let scenario = ScenarioSampler::generate(&config, 7).unwrap();

        for left_index in 0..scenario.targets.len() {
            for right_index in (left_index + 1)..scenario.targets.len() {
                let distance = scenario.targets[left_index]
                    .position
                    .distance(scenario.targets[right_index].position);
                assert!(distance > config.targets.discovery_radius);
            }
        }
    }

    #[test]
    fn generation_fails_when_world_is_too_small_for_obstacles() {
        let mut config = SimulationConfig::default();
        config.world.width = 5.0;
        config.world.height = 5.0;

        let error = ScenarioSampler::generate(&config, 7).unwrap_err();
        assert!(error.contains("world is too small"));
    }
}
