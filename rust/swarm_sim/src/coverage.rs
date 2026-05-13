//! Coverage grid tracking for swarm area-coverage metrics.

use crate::{agent::Agent, config::SimulationConfig, geometry::Vec2};

/// Fixed grid that records which world cells have been sensed by any agent.
#[derive(Clone, Debug)]
pub struct CoverageGrid {
    grid_width: usize,
    grid_height: usize,
    covered: Vec<bool>,
}

impl CoverageGrid {
    /// Creates an empty coverage grid.
    pub fn new(grid_width: usize, grid_height: usize) -> Self {
        Self {
            grid_width,
            grid_height,
            covered: vec![false; grid_width * grid_height],
        }
    }

    /// Clears all covered-cell flags.
    pub fn reset(&mut self) {
        self.covered.fill(false);
    }

    /// Marks cells within coverage sensing radius of any agent and returns newly covered cells.
    pub fn mark_from_agents(&mut self, agents: &[Agent], config: &SimulationConfig) -> usize {
        let mut newly_covered = 0;
        let cell_width = config.world.width / self.grid_width as f64;
        let cell_height = config.world.height / self.grid_height as f64;

        for y in 0..self.grid_height {
            for x in 0..self.grid_width {
                let index = self.index(x, y);
                if self.covered[index] {
                    continue;
                }

                let center = Vec2::new(
                    (x as f64 + 0.5) * cell_width,
                    (y as f64 + 0.5) * cell_height,
                );

                if agents
                    .iter()
                    .any(|agent| agent.position.distance(center) <= config.coverage.sensing_radius)
                {
                    self.covered[index] = true;
                    newly_covered += 1;
                }
            }
        }

        newly_covered
    }

    /// Returns the fraction of covered cells in `[0.0, 1.0]`.
    pub fn ratio(&self) -> f64 {
        if self.covered.is_empty() {
            return 0.0;
        }

        self.covered.iter().filter(|covered| **covered).count() as f64 / self.covered.len() as f64
    }

    /// Returns the number of covered cells.
    pub fn covered_cells(&self) -> usize {
        self.covered.iter().filter(|covered| **covered).count()
    }

    /// Returns the total number of cells in the grid.
    pub fn total_cells(&self) -> usize {
        self.covered.len()
    }

    fn index(&self, x: usize, y: usize) -> usize {
        y * self.grid_width + x
    }
}

#[cfg(test)]
mod tests {
    use crate::{agent::Agent, config::SimulationConfig, geometry::Vec2};

    use super::CoverageGrid;

    #[test]
    fn coverage_ratio_stays_bounded() {
        let config = SimulationConfig::default();
        let agents = vec![Agent::new(0, Vec2::new(50.0, 50.0))];
        let mut grid = CoverageGrid::new(config.coverage.grid_width, config.coverage.grid_height);

        grid.mark_from_agents(&agents, &config);

        assert!((0.0..=1.0).contains(&grid.ratio()));
    }
}
