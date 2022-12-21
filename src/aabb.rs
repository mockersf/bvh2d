use std::f32;
use std::fmt::Display;

use crate::{Point2, Vector2};

use crate::axis::Axis;

#[derive(Copy, Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct AABB {
    pub min: Point2,
    pub max: Point2,
}

impl Display for AABB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "min bound: {}; max bound: {}", self.min, self.max)
    }
}

pub trait Bounded {
    fn aabb(&self) -> AABB;
}

impl AABB {
    #[inline]
    pub const fn with_bounds(min: Point2, max: Point2) -> AABB {
        AABB { min, max }
    }

    pub(crate) const EMPTY: AABB = {
        AABB {
            min: Point2::new(f32::INFINITY, f32::INFINITY),
            max: Point2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    };

    #[inline]
    pub(crate) fn join(&self, other: &AABB) -> AABB {
        AABB::with_bounds(
            Point2::min(self.min, other.min),
            Point2::max(self.max, other.max),
        )
    }

    #[inline]
    pub(crate) fn join_mut(&mut self, other: &AABB) {
        self.min = Point2::min(self.min, other.min);
        self.max = Point2::max(self.max, other.max);
    }

    #[inline]
    pub(crate) fn grow(&self, other: &Point2) -> AABB {
        AABB::with_bounds(Point2::min(self.min, *other), Point2::max(self.max, *other))
    }

    #[inline]
    pub(crate) fn size(&self) -> Vector2 {
        self.max - self.min
    }

    #[inline]
    pub(crate) fn center(&self) -> Point2 {
        self.min + (self.size() / 2.0)
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y
    }

    #[inline]
    pub(crate) fn surface_area(&self) -> f32 {
        let size = self.size();
        size.x * size.y
    }

    #[inline]
    pub(crate) fn largest_axis(&self) -> Axis {
        let size = self.size();
        if size.x > size.y {
            Axis::X
        } else {
            Axis::Y
        }
    }

    pub(crate) fn approx_contains_eps(&self, p: &Point2, epsilon: f32) -> bool {
        (p.x - self.min.x) > -epsilon
            && (p.x - self.max.x) < epsilon
            && (p.y - self.min.y) > -epsilon
            && (p.y - self.max.y) < epsilon
    }

    pub fn approx_contains_aabb_eps(&self, other: &AABB, epsilon: f32) -> bool {
        self.approx_contains_eps(&other.min, epsilon)
            && self.approx_contains_eps(&other.max, epsilon)
    }
}
