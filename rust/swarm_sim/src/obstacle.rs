//! Circular no-fly obstacle state.

use crate::geometry::Vec2;

/// Circular obstacle used for no-fly violation metrics.
#[derive(Clone, Debug)]
pub struct Obstacle {
    /// Stable numeric identifier within the scenario.
    pub id: usize,
    /// Continuous 2D obstacle center.
    pub position: Vec2,
    /// Obstacle radius.
    pub radius: f64,
}

impl Obstacle {
    /// Creates a circular obstacle at `position`.
    pub fn new(id: usize, position: Vec2, radius: f64) -> Self {
        Self {
            id,
            position,
            radius,
        }
    }
}
