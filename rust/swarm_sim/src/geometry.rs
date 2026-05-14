//! Small 2D geometry utilities for continuous world simulation.

/// Continuous 2D vector used for positions, velocities, and actions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    /// X coordinate or component.
    pub x: f64,
    /// Y coordinate or component.
    pub y: f64,
}

impl Vec2 {
    /// Creates a vector from x/y components.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Returns the zero vector.
    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Returns whether both components are finite floating point values.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Returns the Euclidean length.
    pub fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Returns the Euclidean distance between two vectors.
    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }

    /// Returns this vector shortened to `max_length` when it is longer.
    pub fn clamp_length(self, max_length: f64) -> Self {
        let length = self.length();
        if length <= max_length || length == 0.0 {
            self
        } else {
            self * (max_length / length)
        }
    }

    /// Clamps this vector as a point inside `[0, width] x [0, height]`.
    pub fn clamp_to_bounds(self, width: f64, height: f64) -> Self {
        Self {
            x: self.x.clamp(0.0, width),
            y: self.y.clamp(0.0, height),
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

/// Returns true when two circles overlap or touch.
pub fn circles_overlap(center_a: Vec2, radius_a: f64, center_b: Vec2, radius_b: f64) -> bool {
    center_a.distance(center_b) <= radius_a + radius_b
}

#[cfg(test)]
mod tests {
    use super::{circles_overlap, Vec2};

    #[test]
    fn distance_uses_euclidean_norm() {
        assert_eq!(Vec2::new(0.0, 0.0).distance(Vec2::new(3.0, 4.0)), 5.0);
    }

    #[test]
    fn clamp_length_limits_long_vectors() {
        let clamped = Vec2::new(3.0, 4.0).clamp_length(2.0);
        assert!((clamped.length() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn clamp_to_bounds_keeps_coordinates_inside_world() {
        assert_eq!(
            Vec2::new(-1.0, 12.0).clamp_to_bounds(10.0, 8.0),
            Vec2::new(0.0, 8.0)
        );
    }

    #[test]
    fn circle_overlap_includes_touching_edges() {
        assert!(circles_overlap(
            Vec2::new(0.0, 0.0),
            1.0,
            Vec2::new(2.0, 0.0),
            1.0
        ));
    }
}
