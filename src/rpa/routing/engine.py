"""Engine: a loaded graph with everything route generation needs around it."""

from pathlib import Path

import numpy as np

import rpa_kernel
from rpa.graph.snap import Snapper
from rpa.graph.store import GraphData, load_graph
from rpa.routing.cost import Prefs, arc_weights


RIDES_FILE = "rides.npy"


def load_engine(directory: Path) -> "Engine":
    """Engine for the graph in `directory`, with ride counts if tracks were matched to it."""
    graph, _ = load_graph(directory)
    rides_path = Path(directory) / RIDES_FILE
    return Engine(graph, np.load(rides_path) if rides_path.exists() else None)


class Engine:
    """Holds the graph, the search kernel and per-graph lookups. Build once, share across requests."""

    def __init__(self, g: GraphData, rides: np.ndarray | None = None) -> None:
        if rides is not None and len(rides) != g.n_edges:
            raise ValueError(f"rides has {len(rides)} entries but the graph has {g.n_edges} edges")
        self.g = g
        self.rides = rides  # per edge: how many recorded rides used it; None if no tracks were matched
        self.kernel = rpa_kernel.Graph(
            np.ascontiguousarray(g.edge_u), np.ascontiguousarray(g.edge_v), np.ascontiguousarray(g.node_lonlat)
        )
        self.snapper = Snapper(g)
        self.length = np.ascontiguousarray(g.edge_len_m)
        # Arc 2e runs u -> v, arc 2e + 1 runs v -> u.
        self.arc_tail = np.stack([g.edge_u, g.edge_v], axis=1).ravel()
        self.arc_head = np.stack([g.edge_v, g.edge_u], axis=1).ravel()
        self.degree = np.bincount(self.arc_tail, minlength=g.n_nodes)

    def weights(self, prefs: Prefs) -> np.ndarray:
        return arc_weights(self.g, prefs, self.rides)

    def node_niceness(self, w: np.ndarray) -> np.ndarray:
        """Lowest cost multiplier among the arcs leaving each node (1.0 = an ideal road), inf if none.

        Used to place via points on roads the rider would like, not merely on the nearest ones.
        """
        multiplier = w / np.repeat(self.length, 2)
        nice = np.full(self.g.n_nodes, np.inf, dtype=np.float32)
        usable = np.isfinite(multiplier)
        np.minimum.at(nice, self.arc_tail[usable], multiplier[usable])
        return nice
