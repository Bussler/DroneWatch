//! Agent state for homogeneous drones.

use crate::geometry::Vec2;

/// Mutable state for one drone agent.
#[derive(Clone, Debug)]
pub struct Agent {
    /// Stable numeric identifier within the scenario.
    pub id: usize,
    /// Current continuous 2D position.
    pub position: Vec2,
    /// Current velocity inferred from the latest displacement.
    pub velocity: Vec2,
}

impl Agent {
    /// Creates an agent at `position` with zero velocity.
    pub fn new(id: usize, position: Vec2) -> Self {
        Self {
            id,
            position,
            velocity: Vec2::zero(),
        }
    }
}
