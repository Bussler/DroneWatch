use crate::geometry::Vec2;

#[derive(Clone, Debug)]
pub struct Target {
    pub id: usize,
    pub position: Vec2,
    pub discovered: bool,
}

impl Target {
    pub fn new(id: usize, position: Vec2) -> Self {
        Self {
            id,
            position,
            discovered: false,
        }
    }
}
