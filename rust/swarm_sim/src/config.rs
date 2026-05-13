//! Configuration types and validation for the Rust simulation core.

/// String-backed result type used by the Phase 1 simulator API.
pub type SimResult<T> = Result<T, String>;

/// Continuous world bounds and timestep settings.
#[derive(Clone, Copy, Debug)]
pub struct WorldConfig {
    /// World width in continuous coordinate units.
    pub width: f64,
    /// World height in continuous coordinate units.
    pub height: f64,
    /// Simulation timestep used to scale action-controlled movement.
    pub dt: f64,
}

/// Homogeneous agent settings for the default swarm.
#[derive(Clone, Copy, Debug)]
pub struct AgentConfig {
    /// Number of agents spawned on reset.
    pub count: usize,
    /// Maximum distance an agent can move per unit of `dt`.
    pub max_speed: f64,
    /// Radius used for agent-agent collision checks.
    pub collision_radius: f64,
    /// Local sensing radius reserved for observation-building phases.
    pub sensing_radius: f64,
    /// Distance threshold for communication graph edges.
    pub communication_radius: f64,
}

/// Static target settings.
#[derive(Clone, Copy, Debug)]
pub struct TargetConfig {
    /// Number of targets spawned on reset.
    pub count: usize,
    /// Distance at which an agent discovers a target.
    pub discovery_radius: f64,
}

/// Circular no-fly obstacle settings.
#[derive(Clone, Copy, Debug)]
pub struct ObstacleConfig {
    /// Number of circular obstacles spawned on reset.
    pub count: usize,
    /// Minimum obstacle radius sampled during scenario generation.
    pub min_radius: f64,
    /// Maximum obstacle radius sampled during scenario generation.
    pub max_radius: f64,
}

/// Coverage grid settings used by the simulator metrics.
#[derive(Clone, Copy, Debug)]
pub struct CoverageConfig {
    /// Number of grid cells along the x-axis.
    pub grid_width: usize,
    /// Number of grid cells along the y-axis.
    pub grid_height: usize,
    /// Radius used to mark grid cells covered by each agent.
    pub sensing_radius: f64,
}

/// Full simulator configuration for the Phase 1 Rust core.
#[derive(Clone, Copy, Debug)]
pub struct SimulationConfig {
    /// Maximum number of successful steps in one episode.
    pub max_episode_steps: usize,
    /// Continuous world settings.
    pub world: WorldConfig,
    /// Homogeneous swarm settings.
    pub agents: AgentConfig,
    /// Static target settings.
    pub targets: TargetConfig,
    /// Circular obstacle settings.
    pub obstacles: ObstacleConfig,
    /// Coverage grid settings.
    pub coverage: CoverageConfig,
}

impl SimulationConfig {
    /// Validates that the configuration can drive a finite 2D simulation.
    pub fn validate(&self) -> SimResult<()> {
        if self.max_episode_steps == 0 {
            return Err("max_episode_steps must be greater than zero".to_string());
        }
        if self.agents.count == 0 {
            return Err("agent count must be greater than zero".to_string());
        }
        if self.world.width <= 0.0 || self.world.height <= 0.0 || self.world.dt <= 0.0 {
            return Err("world width, height, and dt must be positive".to_string());
        }
        if self.agents.max_speed <= 0.0
            || self.agents.collision_radius <= 0.0
            || self.agents.communication_radius <= 0.0
            || self.agents.sensing_radius <= 0.0
        {
            return Err("agent radii and max speed must be positive".to_string());
        }
        if self.targets.discovery_radius <= 0.0 {
            return Err("target discovery radius must be positive".to_string());
        }
        if self.obstacles.min_radius <= 0.0 || self.obstacles.max_radius < self.obstacles.min_radius
        {
            return Err("obstacle radii must be positive and ordered".to_string());
        }
        if self.coverage.grid_width == 0
            || self.coverage.grid_height == 0
            || self.coverage.sensing_radius <= 0.0
        {
            return Err("coverage grid dimensions and sensing radius must be positive".to_string());
        }
        Ok(())
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: 200,
            world: WorldConfig {
                width: 100.0,
                height: 100.0,
                dt: 1.0,
            },
            agents: AgentConfig {
                count: 16,
                max_speed: 2.0,
                collision_radius: 0.75,
                sensing_radius: 15.0,
                communication_radius: 20.0,
            },
            targets: TargetConfig {
                count: 20,
                discovery_radius: 2.0,
            },
            obstacles: ObstacleConfig {
                count: 8,
                min_radius: 2.0,
                max_radius: 6.0,
            },
            coverage: CoverageConfig {
                grid_width: 50,
                grid_height: 50,
                sensing_radius: 10.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SimulationConfig;

    #[test]
    fn default_config_is_valid() {
        SimulationConfig::default().validate().unwrap();
    }

    #[test]
    fn rejects_zero_episode_horizon() {
        let mut config = SimulationConfig::default();
        config.max_episode_steps = 0;

        assert!(config.validate().unwrap_err().contains("max_episode_steps"));
    }

    #[test]
    fn rejects_zero_agents() {
        let mut config = SimulationConfig::default();
        config.agents.count = 0;

        assert!(config.validate().unwrap_err().contains("agent count"));
    }

    #[test]
    fn rejects_invalid_world_dimensions() {
        let mut config = SimulationConfig::default();
        config.world.width = 0.0;

        assert!(config.validate().unwrap_err().contains("world width"));
    }

    #[test]
    fn rejects_invalid_obstacle_radius_range() {
        let mut config = SimulationConfig::default();
        config.obstacles.min_radius = 5.0;
        config.obstacles.max_radius = 2.0;

        assert!(config.validate().unwrap_err().contains("obstacle radii"));
    }

    #[test]
    fn rejects_invalid_coverage_grid() {
        let mut config = SimulationConfig::default();
        config.coverage.grid_width = 0;

        assert!(config.validate().unwrap_err().contains("coverage grid"));
    }
}
