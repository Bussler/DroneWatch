//! Static target state and discovery flag.

use crate::geometry::Vec2;

/// Target that can be discovered once by any nearby agent.
#[derive(Clone, Debug)]
pub struct Target {
    /// Stable numeric identifier within the scenario.
    pub id: usize,
    /// Continuous 2D target position.
    pub position: Vec2,
    /// Whether this target has already been discovered.
    pub discovered: bool,
}

impl Target {
    /// Creates an undiscovered target at `position`.
    pub fn new(id: usize, position: Vec2) -> Self {
        Self {
            id,
            position,
            discovered: false,
        }
    }
}
