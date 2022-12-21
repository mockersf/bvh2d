use crate::aabb::{Bounded, AABB};
use crate::bvh2d::iter::BVH2dTraverseIterator;
use crate::utils::{joint_aabb_of_shapes, Bucket};
use crate::Point2;
use crate::EPSILON;
use std::f32;
use std::mem::MaybeUninit;

#[derive(Copy, Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum BVH2dNode {
    Leaf {
        parent_index: usize,
        shape_index: usize,
    },

    Node {
        parent_index: usize,
        child_l_index: usize,
        child_l_aabb: AABB,
        child_r_index: usize,
        child_r_aabb: AABB,
    },
}

impl PartialEq for BVH2dNode {
    // TODO Consider also comparing AABBs
    fn eq(&self, other: &BVH2dNode) -> bool {
        match (self, other) {
            (
                &BVH2dNode::Node {
                    parent_index: self_parent_index,
                    child_l_index: self_child_l_index,
                    child_r_index: self_child_r_index,
                    ..
                },
                &BVH2dNode::Node {
                    parent_index: other_parent_index,
                    child_l_index: other_child_l_index,
                    child_r_index: other_child_r_index,
                    ..
                },
            ) => {
                self_parent_index == other_parent_index
                    && self_child_l_index == other_child_l_index
                    && self_child_r_index == other_child_r_index
            }
            (
                &BVH2dNode::Leaf {
                    parent_index: self_parent_index,
                    shape_index: self_shape_index,
                },
                &BVH2dNode::Leaf {
                    parent_index: other_parent_index,
                    shape_index: other_shape_index,
                },
            ) => self_parent_index == other_parent_index && self_shape_index == other_shape_index,
            _ => false,
        }
    }
}

impl BVH2dNode {
    /// Returns the index of the shape contained within the node if is a leaf,
    /// or `None` if it is an interior node.
    #[allow(unused)]
    pub fn shape_index(&self) -> Option<usize> {
        match *self {
            BVH2dNode::Leaf { shape_index, .. } => Some(shape_index),
            _ => None,
        }
    }

    /// Returns the index of the parent node.
    pub fn parent(&self) -> usize {
        match *self {
            BVH2dNode::Node { parent_index, .. } | BVH2dNode::Leaf { parent_index, .. } => {
                parent_index
            }
        }
    }

    /// Returns the depth of the node. The root node has depth `0`.
    pub fn depth(&self, nodes: &[BVH2dNode]) -> u32 {
        let parent_i = self.parent();
        if parent_i == 0 && nodes[parent_i].eq(self) {
            0
        } else {
            1 + nodes[parent_i].depth(nodes)
        }
    }

    fn build<T: Bounded + Send + Sync>(
        shapes: &[T],
        indices: &mut [usize],
        nodes: &mut [MaybeUninit<BVH2dNode>],
        node_index: usize,
        parent_index: usize,
    ) {
        // Helper function to accumulate the AABB joint and the centroids AABB
        #[inline]
        fn grow_convex_hull(convex_hull: (AABB, AABB), shape_aabb: &AABB) -> (AABB, AABB) {
            let center = &shape_aabb.center();
            let convex_hull_aabbs = &convex_hull.0;
            let convex_hull_centroids = &convex_hull.1;
            (
                convex_hull_aabbs.join(shape_aabb),
                convex_hull_centroids.grow(center),
            )
        }

        // If there is only one element left, don't split anymore
        if indices.len() == 1 {
            let shape_index = indices[0];
            nodes[0].write(BVH2dNode::Leaf {
                parent_index,
                shape_index,
            });
            // Let the shape know the index of the node that represents it.
            return;
        }

        #[allow(unused_mut)]
        let mut parallel_recurse = false;
        #[cfg(feature = "rayon")]
        if indices.len() > 64 {
            parallel_recurse = true;
        }

        let mut convex_hull = (AABB::EMPTY, AABB::EMPTY);
        for index in indices.iter() {
            convex_hull = grow_convex_hull(convex_hull, &shapes[*index].aabb());
        }
        let (aabb_bounds, centroid_bounds) = convex_hull;

        // Find the axis along which the shapes are spread the most.
        let split_axis = centroid_bounds.largest_axis();
        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];

        // The following `if` partitions `indices` for recursively calling `BVH2d::build`.
        let (child_l_index, child_l_aabb, child_r_index, child_r_aabb) = if split_axis_size
            < EPSILON
        {
            // In this branch the shapes lie too close together so that splitting them in a
            // sensible way is not possible. Instead we just split the list of shapes in half.
            let (child_l_indices, child_r_indices) = indices.split_at_mut(indices.len() / 2);
            let child_l_aabb = joint_aabb_of_shapes(child_l_indices, shapes);
            let child_r_aabb = joint_aabb_of_shapes(child_r_indices, shapes);

            let next_nodes = &mut nodes[1..];
            let (l_nodes, r_nodes) = next_nodes.split_at_mut(child_l_indices.len() * 2 - 1);
            let child_l_index = node_index + 1;
            let child_r_index = node_index + 1 + l_nodes.len();

            // Proceed recursively.
            if parallel_recurse {
                // parallel_recurse is only ever true when the rayon feature is enabled
                #[cfg(feature = "rayon")]
                {
                    rayon::join(
                        || {
                            BVH2dNode::build(
                                shapes,
                                child_l_indices,
                                l_nodes,
                                child_l_index, // The new node's index
                                node_index,    // The parent index
                            )
                        },
                        || {
                            BVH2dNode::build(
                                shapes,
                                child_r_indices,
                                r_nodes,
                                child_r_index, // The new node's index
                                node_index,    // The parent index
                            )
                        },
                    );
                }
            } else {
                BVH2dNode::build(shapes, child_l_indices, l_nodes, child_l_index, node_index);
                BVH2dNode::build(shapes, child_r_indices, r_nodes, child_r_index, node_index);
            }

            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        } else {
            const NUM_BUCKETS: usize = 4;
            const BUCKET_START_CAPACITY: usize = 20;
            let mut buckets = [Bucket::EMPTY; NUM_BUCKETS];
            let mut bucket_assignments: [Vec<usize>; NUM_BUCKETS] = [
                Vec::with_capacity(BUCKET_START_CAPACITY),
                Vec::with_capacity(BUCKET_START_CAPACITY),
                Vec::with_capacity(BUCKET_START_CAPACITY),
                Vec::with_capacity(BUCKET_START_CAPACITY),
            ];

            // In this branch the `split_axis_size` is large enough to perform meaningful splits.
            // We start by assigning the shapes to `Bucket`s.
            for idx in indices.iter() {
                let shape = &shapes[*idx];
                let shape_aabb = shape.aabb();
                let shape_center = shape_aabb.center();

                // Get the relative position of the shape centroid `[0.0..1.0]`.
                let bucket_num_relative =
                    (shape_center[split_axis] - centroid_bounds.min[split_axis]) / split_axis_size;

                // Convert that to the actual `Bucket` number.
                let bucket_num = (bucket_num_relative * (NUM_BUCKETS as f32 - 0.01)) as usize;

                // Extend the selected `Bucket` and add the index to the actual bucket.
                buckets[bucket_num].add_aabb(&shape_aabb);
                bucket_assignments[bucket_num].push(*idx);
            }

            // Compute the costs for each configuration and select the best configuration.
            let mut min_bucket = 0;
            let mut min_cost = f32::INFINITY;
            let mut child_l_aabb = AABB::EMPTY;
            let mut child_r_aabb = AABB::EMPTY;
            for i in 0..(NUM_BUCKETS - 1) {
                let (l_buckets, r_buckets) = buckets.split_at(i + 1);
                let child_l = l_buckets.iter().fold(Bucket::EMPTY, Bucket::join_bucket);
                let child_r = r_buckets.iter().fold(Bucket::EMPTY, Bucket::join_bucket);

                let cost = (child_l.size as f32 * child_l.aabb.surface_area()
                    + child_r.size as f32 * child_r.aabb.surface_area())
                    / aabb_bounds.surface_area();
                if cost < min_cost {
                    min_bucket = i;
                    min_cost = cost;
                    child_l_aabb = child_l.aabb;
                    child_r_aabb = child_r.aabb;
                }
            }

            // Join together all index buckets.
            let (l_assignments, r_assignments) = bucket_assignments.split_at_mut(min_bucket + 1);

            let mut l_count = 0;
            for group in l_assignments.iter() {
                l_count += group.len();
            }

            let (child_l_indices, child_r_indices) = indices.split_at_mut(l_count);

            let mut i = 0;
            for group in l_assignments.iter() {
                for x in group {
                    child_l_indices[i] = *x;
                    i += 1;
                }
            }
            i = 0;
            for group in r_assignments.iter() {
                for x in group {
                    child_r_indices[i] = *x;
                    i += 1;
                }
            }

            let next_nodes = &mut nodes[1..];
            let (l_nodes, r_nodes) = next_nodes.split_at_mut(child_l_indices.len() * 2 - 1);

            let child_l_index = node_index + 1;
            let child_r_index = node_index + 1 + l_nodes.len();

            // Proceed recursively.
            if parallel_recurse {
                // parallel_recurse is only ever true when the rayon feature is enabled
                #[cfg(feature = "rayon")]
                {
                    rayon::join(
                        || {
                            BVH2dNode::build(
                                shapes,
                                child_l_indices,
                                l_nodes,
                                child_l_index, // The new node's index
                                node_index,    // The parent index
                            )
                        },
                        || {
                            BVH2dNode::build(
                                shapes,
                                child_r_indices,
                                r_nodes,
                                child_r_index, // The new node's index
                                node_index,    // The parent index
                            )
                        },
                    );
                }
            } else {
                BVH2dNode::build(shapes, child_l_indices, l_nodes, child_l_index, node_index);
                BVH2dNode::build(shapes, child_r_indices, r_nodes, child_r_index, node_index);
            }

            (child_l_index, child_l_aabb, child_r_index, child_r_aabb)
        };

        // Construct the actual data structure and replace the dummy node.
        debug_assert!(!child_l_aabb.is_empty());
        debug_assert!(!child_r_aabb.is_empty());
        nodes[0].write(BVH2dNode::Node {
            parent_index,
            child_l_aabb,
            child_l_index,
            child_r_aabb,
            child_r_index,
        });
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct BVH2d {
    pub(crate) nodes: Vec<BVH2dNode>,
}

impl BVH2d {
    pub fn build<Shape: Bounded + Send + Sync>(shapes: &[Shape]) -> BVH2d {
        if shapes.is_empty() {
            return BVH2d { nodes: Vec::new() };
        }

        let mut indices = (0..shapes.len()).collect::<Vec<usize>>();
        let expected_node_count = shapes.len() * 2 - 1;
        let mut nodes = Vec::with_capacity(expected_node_count);

        let uninit_slice = unsafe {
            std::slice::from_raw_parts_mut(
                nodes.as_mut_ptr() as *mut MaybeUninit<BVH2dNode>,
                expected_node_count,
            )
        };

        BVH2dNode::build(shapes, &mut indices, uninit_slice, 0, 0);

        unsafe {
            nodes.set_len(expected_node_count);
        }

        BVH2d { nodes }
    }

    pub fn contains_iterator<'a>(&'a self, point: &'a Point2) -> BVH2dTraverseIterator {
        BVH2dTraverseIterator::new(self, point)
    }

    // /// Prints the [`BVH`] in a tree-like visualization.
    // ///
    // /// [`BVH`]: struct.BVH.html
    // ///
    pub fn pretty_print(&self) {
        self.print_node(0);
    }

    fn print_node(&self, node_index: usize) {
        let nodes = &self.nodes;
        match nodes[node_index] {
            BVH2dNode::Node {
                child_l_index,
                child_r_index,
                child_l_aabb,
                child_r_aabb,
                ..
            } => {
                let depth = nodes[node_index].depth(nodes);
                let padding: String = " ".repeat(depth as usize);
                println!(
                    "{padding}node={node_index} parent={}",
                    nodes[node_index].parent()
                );
                println!("{padding}{child_l_index} child_l {child_l_aabb}");
                self.print_node(child_l_index);
                println!("{padding}{child_r_index} child_r {child_r_aabb}");
                self.print_node(child_r_index);
            }
            BVH2dNode::Leaf { shape_index, .. } => {
                let depth = nodes[node_index].depth(nodes);
                let padding: String = " ".repeat(depth as usize);
                println!(
                    "{padding}node={node_index} parent={}",
                    nodes[node_index].parent()
                );
                println!("{padding}shape\t{shape_index:?}");
            }
        }
    }

    /// Verifies that the node at index `node_index` lies inside `expected_outer_aabb`,
    /// its parent index is equal to `expected_parent_index`, its depth is equal to
    /// `expected_depth`. Increares `node_count` by the number of visited nodes.
    fn is_consistent_subtree<Shape: Bounded>(
        &self,
        node_index: usize,
        expected_parent_index: usize,
        expected_outer_aabb: &AABB,
        expected_depth: u32,
        node_count: &mut usize,
        shapes: &[Shape],
    ) -> bool {
        *node_count += 1;
        match self.nodes[node_index] {
            BVH2dNode::Node {
                parent_index,
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
            } => {
                let depth = self.nodes[node_index].depth(self.nodes.as_slice());
                let correct_parent_index = expected_parent_index == parent_index;
                let correct_depth = expected_depth == depth;
                let left_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&child_l_aabb, EPSILON);
                let right_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&child_r_aabb, EPSILON);
                let left_subtree_consistent = self.is_consistent_subtree(
                    child_l_index,
                    node_index,
                    &child_l_aabb,
                    expected_depth + 1,
                    node_count,
                    shapes,
                );
                let right_subtree_consistent = self.is_consistent_subtree(
                    child_r_index,
                    node_index,
                    &child_r_aabb,
                    expected_depth + 1,
                    node_count,
                    shapes,
                );

                correct_parent_index
                    && correct_depth
                    && left_aabb_in_parent
                    && right_aabb_in_parent
                    && left_subtree_consistent
                    && right_subtree_consistent
            }
            BVH2dNode::Leaf {
                parent_index,
                shape_index,
            } => {
                let depth = self.nodes[node_index].depth(self.nodes.as_slice());
                let correct_parent_index = expected_parent_index == parent_index;
                let correct_depth = expected_depth == depth;
                let shape_aabb = shapes[shape_index].aabb();
                let shape_aabb_in_parent =
                    expected_outer_aabb.approx_contains_aabb_eps(&shape_aabb, EPSILON);

                correct_parent_index && correct_depth && shape_aabb_in_parent
            }
        }
    }

    /// Checks if all children of a node have the correct parent index, and that there is no
    /// detached subtree. Also checks if the `AABB` hierarchy is consistent.
    pub fn is_consistent<Shape: Bounded + Send + Sync>(&self, shapes: &[Shape]) -> bool {
        // The root node of the bvh is not bounded by anything.
        let space = AABB {
            min: Point2::new(f32::NEG_INFINITY, f32::NEG_INFINITY),
            max: Point2::new(f32::INFINITY, f32::INFINITY),
        };

        // The counter for all nodes.
        let mut node_count = 0;
        let subtree_consistent =
            self.is_consistent_subtree(0, 0, &space, 0, &mut node_count, shapes);

        // Check if all nodes have been counted from the root node.
        // If this is false, it means we have a detached subtree.
        let is_connected = node_count == self.nodes.len();
        subtree_consistent && is_connected
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        aabb::{Bounded, AABB},
        bvh2d::BVH2dNode,
        Point2, Vector2,
    };

    use super::BVH2d;

    /// Define some `Bounded` structure.
    #[derive(Debug, Clone, Copy)]
    #[cfg_attr(feature = "serde_impls", derive(serde::Serialize, serde::Deserialize))]
    pub struct UnitBox {
        pub id: i32,
        pub pos: Point2,
        _node_index: usize,
    }

    impl UnitBox {
        pub fn new(id: i32, pos: Point2) -> UnitBox {
            UnitBox {
                id,
                pos,
                _node_index: 0,
            }
        }
    }

    /// `UnitBox`'s `AABB`s are unit `AABB`s centered on the box's position.
    impl Bounded for UnitBox {
        fn aabb(&self) -> AABB {
            let min = self.pos + Vector2::new(-0.5, -0.5);
            let max = self.pos + Vector2::new(0.5, 0.5);
            AABB::with_bounds(min, max)
        }
    }

    /// Generate 21 `UnitBox`s along the X axis centered on whole numbers (-10,9,..,10).
    /// The index is set to the rounded x-coordinate of the box center.
    pub fn generate_aligned_boxes() -> Vec<UnitBox> {
        // Create 21 boxes along the x-axis
        let mut shapes = Vec::new();
        for x in -10..11 {
            shapes.push(UnitBox::new(x, Point2::new(x as f32, 0.0)));
        }
        shapes
    }

    /// Creates a `BoundingHierarchy` for a fixed scene structure.
    pub fn build_some_bh() -> (Vec<UnitBox>, BVH2d) {
        let boxes = generate_aligned_boxes();
        let bh = BVH2d::build(&boxes);
        (boxes, bh)
    }

    #[test]
    /// Tests whether the building procedure succeeds in not failing.
    fn test_build_bvh() {
        let (shapes, bvh) = build_some_bh();
        bvh.pretty_print();
        bvh.is_consistent(shapes.as_slice());
    }

    #[test]
    /// Verify contents of the bounding hierarchy for a fixed scene structure
    fn test_bvh_shape_indices() {
        use std::collections::HashSet;

        let (all_shapes, bh) = build_some_bh();

        // It should find all shape indices.
        let expected_shapes: HashSet<_> = (0..all_shapes.len()).collect();
        let mut found_shapes = HashSet::new();

        for node in bh.nodes.iter() {
            match *node {
                BVH2dNode::Node { .. } => {
                    assert_eq!(node.shape_index(), None);
                }
                BVH2dNode::Leaf { .. } => {
                    found_shapes.insert(
                        node.shape_index()
                            .expect("getting a shape index from a leaf node"),
                    );
                }
            }
        }

        assert_eq!(expected_shapes, found_shapes);
    }
}
