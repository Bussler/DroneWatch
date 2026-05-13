use crate::geometry::Vec2;

#[derive(Clone, Debug)]
pub struct Obstacle {
    pub id: usize,
    pub position: Vec2,
    pub radius: f64,
}

impl Obstacle {
    pub fn new(id: usize, position: Vec2, radius: f64) -> Self {
        Self {
            id,
            position,
            radius,
        }
    }
}
