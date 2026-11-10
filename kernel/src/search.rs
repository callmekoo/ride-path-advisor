//! Path search over a compact road graph. No Python types in this module.
//!
//! Arc numbering: edge `e` yields arc `2e` (u -> v) and arc `2e + 1` (v -> u),
//! so `reverse(a) = a ^ 1` and `edge(a) = a >> 1`.
//!
//! Weight contract: `w[a]` is either non-finite (the arc is forbidden) or at
//! least the length in metres of its edge. That keeps the straight-line
//! heuristic of `route` admissible.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub const NO_ARC: u32 = u32::MAX;
const EARTH_RADIUS_M: f64 = 6_371_008.8;

pub struct Csr {
    n_nodes: usize,
    n_edges: usize,
    /// Arcs leaving node `n` are `out_arc[out_off[n]..out_off[n + 1]]`.
    out_off: Vec<u32>,
    out_arc: Vec<u32>,
    /// Node an arc points to. Its tail is `head[a ^ 1]`.
    head: Vec<u32>,
    /// Unit-sphere coordinates of the nodes.
    xyz: Vec<[f64; 3]>,
}

/// Heap entry ordered so that `BinaryHeap` pops the smallest key first.
#[derive(PartialEq)]
struct Item {
    key: f64,
    g: f64,
    node: u32,
}

impl Eq for Item {}

impl Ord for Item {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .key
            .total_cmp(&self.key)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for Item {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-thread buffers for `route`. Entries are valid only when their stamp
/// equals the current epoch, so starting a query costs O(1), not O(nodes).
struct Scratch {
    stamp: Vec<u32>,
    dist: Vec<f64>,
    pred: Vec<u32>,
    heap: BinaryHeap<Item>,
    epoch: u32,
}

impl Scratch {
    fn begin(&mut self, n: usize) {
        if self.stamp.len() != n {
            self.stamp = vec![0; n];
            self.dist = vec![0.0; n];
            self.pred = vec![NO_ARC; n];
            self.epoch = 0;
        }
        self.heap.clear();
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.fill(0);
            self.epoch = 1;
        }
    }

    #[inline]
    fn get(&self, node: usize) -> f64 {
        if self.stamp[node] == self.epoch {
            self.dist[node]
        } else {
            f64::INFINITY
        }
    }

    #[inline]
    fn set(&mut self, node: usize, dist: f64, pred: u32) {
        self.stamp[node] = self.epoch;
        self.dist[node] = dist;
        self.pred[node] = pred;
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const {
        RefCell::new(Scratch {
            stamp: Vec::new(),
            dist: Vec::new(),
            pred: Vec::new(),
            heap: BinaryHeap::new(),
            epoch: 0,
        })
    };
}

/// Straight-line distance through the sphere, in metres. Never longer than
/// any path on the surface between the same two points.
#[inline]
fn chord_m(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt() * EARTH_RADIUS_M
}

impl Csr {
    pub fn new(edge_u: &[u32], edge_v: &[u32], lonlat: &[[f64; 2]]) -> Result<Self, String> {
        if edge_u.len() != edge_v.len() {
            return Err(format!(
                "edge_u has {} entries but edge_v has {}",
                edge_u.len(),
                edge_v.len()
            ));
        }
        let n_nodes = lonlat.len();
        let n_edges = edge_u.len();
        if 2 * n_edges >= NO_ARC as usize {
            return Err(format!("too many edges: {n_edges}"));
        }

        let mut out_off = vec![0u32; n_nodes + 1];
        for (e, (&u, &v)) in edge_u.iter().zip(edge_v).enumerate() {
            if u as usize >= n_nodes || v as usize >= n_nodes {
                return Err(format!(
                    "edge {e} joins nodes {u} and {v}, but there are only {n_nodes} nodes"
                ));
            }
            out_off[u as usize + 1] += 1;
            out_off[v as usize + 1] += 1;
        }
        for n in 0..n_nodes {
            out_off[n + 1] += out_off[n];
        }

        let mut cursor = out_off[..n_nodes].to_vec();
        let mut out_arc = vec![0u32; 2 * n_edges];
        let mut head = vec![0u32; 2 * n_edges];
        for (e, (&u, &v)) in edge_u.iter().zip(edge_v).enumerate() {
            let arc = 2 * e as u32;
            head[arc as usize] = v;
            head[arc as usize + 1] = u;
            out_arc[cursor[u as usize] as usize] = arc;
            cursor[u as usize] += 1;
            out_arc[cursor[v as usize] as usize] = arc + 1;
            cursor[v as usize] += 1;
        }

        let xyz = lonlat
            .iter()
            .map(|&[lon, lat]| {
                let (lon, lat) = (lon.to_radians(), lat.to_radians());
                [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
            })
            .collect();

        Ok(Csr {
            n_nodes,
            n_edges,
            out_off,
            out_arc,
            head,
            xyz,
        })
    }

    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    #[inline]
    fn out_arcs(&self, node: u32) -> &[u32] {
        let lo = self.out_off[node as usize] as usize;
        let hi = self.out_off[node as usize + 1] as usize;
        &self.out_arc[lo..hi]
    }

    /// Cheapest path from `src` to `dst` (A*). Returns the arcs in travel
    /// order and the total cost, or `None` when `dst` cannot be reached.
    pub fn route(&self, src: u32, dst: u32, w: &[f32], h_scale: f64) -> Option<(Vec<u32>, f64)> {
        SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let s = &mut *guard;
            s.begin(self.n_nodes);
            let target = self.xyz[dst as usize];

            s.set(src as usize, 0.0, NO_ARC);
            s.heap.push(Item {
                key: chord_m(&self.xyz[src as usize], &target) * h_scale,
                g: 0.0,
                node: src,
            });

            let mut found = false;
            while let Some(Item { g, node, .. }) = s.heap.pop() {
                if g > s.get(node as usize) {
                    continue; // a cheaper entry for this node was already expanded
                }
                if node == dst {
                    found = true;
                    break;
                }
                for &arc in self.out_arcs(node) {
                    let weight = w[arc as usize];
                    if !weight.is_finite() {
                        continue;
                    }
                    let next = self.head[arc as usize];
                    let next_g = g + weight as f64;
                    if next_g < s.get(next as usize) {
                        s.set(next as usize, next_g, arc);
                        s.heap.push(Item {
                            key: next_g + chord_m(&self.xyz[next as usize], &target) * h_scale,
                            g: next_g,
                            node: next,
                        });
                    }
                }
            }
            if !found {
                return None;
            }

            let cost = s.get(dst as usize);
            let mut arcs = Vec::new();
            let mut node = dst;
            while node != src {
                let arc = s.pred[node as usize];
                arcs.push(arc);
                node = self.head[(arc ^ 1) as usize];
            }
            arcs.reverse();
            Some((arcs, cost))
        })
    }

    /// Cheapest costs from `src` to every node (Dijkstra), or with `reverse`
    /// from every node to `src`. Alongside the cost it sums `aux` (one value
    /// per edge, e.g. metres) along each cheapest path.
    ///
    /// `pred[n]` is the last arc of the path to `n`; with `reverse` it is the
    /// first arc of the path from `n`. Nodes costlier than `max_cost` are
    /// left unreached: cost and aux infinite, `pred` set to `NO_ARC`.
    pub fn sssp(
        &self,
        src: u32,
        w: &[f32],
        aux: &[f32],
        max_cost: f64,
        reverse: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
        let mut dist = vec![f64::INFINITY; self.n_nodes];
        let mut aux_sum = vec![f64::INFINITY; self.n_nodes];
        let mut pred = vec![NO_ARC; self.n_nodes];
        let mut heap = BinaryHeap::new();

        dist[src as usize] = 0.0;
        aux_sum[src as usize] = 0.0;
        heap.push(Item {
            key: 0.0,
            g: 0.0,
            node: src,
        });

        while let Some(Item { g, node, .. }) = heap.pop() {
            if g > dist[node as usize] {
                continue;
            }
            for &out in self.out_arcs(node) {
                // Searching backwards, we travel the arc that enters `node`.
                let arc = if reverse { out ^ 1 } else { out };
                let weight = w[arc as usize];
                if !weight.is_finite() {
                    continue;
                }
                let next = self.head[out as usize] as usize;
                let next_g = g + weight as f64;
                if next_g > max_cost || next_g >= dist[next] {
                    continue;
                }
                dist[next] = next_g;
                aux_sum[next] = aux_sum[node as usize] + aux[(arc >> 1) as usize] as f64;
                pred[next] = arc;
                heap.push(Item {
                    key: next_g,
                    g: next_g,
                    node: next as u32,
                });
            }
        }

        (
            dist.into_iter().map(|d| d as f32).collect(),
            aux_sum.into_iter().map(|d| d as f32).collect(),
            pred,
        )
    }
}
