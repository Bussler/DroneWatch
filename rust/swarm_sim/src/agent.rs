use crate::geometry::Vec2;

#[derive(Clone, Debug)]
pub struct Agent {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
}

impl Agent {
    pub fn new(id: usize, position: Vec2) -> Self {
        Self {
            id,
            position,
            velocity: Vec2::zero(),
        }
    }
}
