#[derive(Clone, Copy, Debug)]
pub struct WorldConfig {
    pub width: f64,
    pub height: f64,
    pub dt: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct AgentConfig {
    pub count: usize,
    pub max_speed: f64,
    pub collision_radius: f64,
    pub sensing_radius: f64,
    pub communication_radius: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct TargetConfig {
    pub count: usize,
    pub discovery_radius: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ObstacleConfig {
    pub count: usize,
    pub min_radius: f64,
    pub max_radius: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct CoverageConfig {
    pub grid_width: usize,
    pub grid_height: usize,
    pub sensing_radius: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SimulationConfig {
    pub max_episode_steps: usize,
    pub world: WorldConfig,
    pub agents: AgentConfig,
    pub targets: TargetConfig,
    pub obstacles: ObstacleConfig,
    pub coverage: CoverageConfig,
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
