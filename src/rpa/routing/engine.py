"""Engine: a loaded graph with everything route generation needs around it."""

from pathlib import Path

import numpy as np

import rpa_kernel
from rpa.arrays import ranges
from rpa.enrich.water import load_water
from rpa.graph.snap import Snapper
from rpa.graph.store import GraphData, load_graph
from rpa.routing.cost import Prefs, arc_weights


RIDES_FILE = "rides.npy"
# Written by `rpa enrich`: forest share, elevation samples along the edges, ascent per direction.
TERRAIN_FILES = ("forest.npy", "prof_off.npy", "prof_ele.npy", "ascent.npy")


def load_engine(directory: Path) -> "Engine":
    """Engine for the graph in `directory`, with ride counts if tracks were matched to it."""
    graph, _ = load_graph(directory)
    rides_path = Path(directory) / RIDES_FILE
    engine = Engine(graph, np.load(rides_path) if rides_path.exists() else None)
    water = load_water(directory)
    if water is not None:
        engine.set_water(*water)
    directory = Path(directory)
    if all((directory / name).exists() for name in TERRAIN_FILES):
        engine.set_terrain(*(np.load(directory / name) for name in TERRAIN_FILES))
    return engine


_WATER_WORD = {"lake": "озеро", "pond": "пруд", "reservoir": "водохранилище", "river": "река", "water": "водоём"}
# Standing water is a better place to ride to than a river crossing.
_RANK_STANDING, _RANK_RIVER = 0.0, 1.0


def _water_label(body: dict) -> str:
    word, name = _WATER_WORD[body["kind"]], body["name"]
    if not name:
        return word
    # Many names already say what they are: "озеро Чусовое", "Верхнемакаровский пруд".
    says_what = any(known in name.lower() for known in ("озеро", "пруд", "водохранилище", "река", "вдхр", "карьер"))
    return name if says_what else f"{word} {name}"


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
        self.forest: np.ndarray | None = None  # f32[E], share of the edge in forest
        self.ascent: np.ndarray | None = None  # f32[E, 2], metres climbed riding u -> v and v -> u
        self._prof_off: np.ndarray | None = None  # edge e owns elevation samples prof_off[e]:prof_off[e + 1]
        self._prof_ele: np.ndarray | None = None  # i16, decimetres, evenly spaced along the edge
        self.at_water: np.ndarray | None = None  # bool[N]; None until water is known
        self.water_rank: np.ndarray | None = None  # f32[N]: how good a water destination, lower is better; inf = none
        self._water_of_node: dict[int, str] = {}

    def set_rides(self, rides: np.ndarray | None) -> None:
        """Replace the ride counts, e.g. after new tracks were matched."""
        if rides is not None and len(rides) != self.g.n_edges:
            raise ValueError(f"rides has {len(rides)} entries but the graph has {self.g.n_edges} edges")
        self.rides = rides

    def set_water(self, nodes: np.ndarray, bodies: list[dict]) -> None:
        """Register water access: `nodes` is u32[K, 2] of (node, index into `bodies`)."""
        self.at_water = np.zeros(self.g.n_nodes, dtype=bool)
        self.at_water[nodes[:, 0]] = True
        rank = np.array([_RANK_RIVER if bodies[body]["kind"] == "river" else _RANK_STANDING for body in nodes[:, 1]])
        self.water_rank = np.full(self.g.n_nodes, np.inf, dtype=np.float32)
        np.minimum.at(self.water_rank, nodes[:, 0], rank)
        # Where a node touches several bodies, name the largest.
        order = np.argsort([bodies[body]["area_m2"] for body in nodes[:, 1]], kind="stable")
        self._water_of_node = {int(nodes[i, 0]): _water_label(bodies[int(nodes[i, 1])]) for i in order}

    def water_label(self, node: int) -> str | None:
        """What water the node is at, e.g. "озеро Шарташ"; None if it is not at water."""
        return self._water_of_node.get(node)

    def set_terrain(self, forest: np.ndarray, prof_off: np.ndarray, prof_ele: np.ndarray, ascent: np.ndarray) -> None:
        """Register what `rpa enrich` sampled: forest as u8 (0..255), elevation profiles, ascent."""
        self.forest = forest.astype(np.float32) / 255
        self.ascent = ascent
        self._prof_off = prof_off.astype(np.int64)
        self._prof_ele = prof_ele

    def profile(self, arcs: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(distance from the start in metres, elevation in metres) along a route; None without elevation data."""
        if self._prof_off is None:
            return None
        if len(arcs) == 0:
            return np.empty(0), np.empty(0)
        edges = (arcs >> 1).astype(np.int64)
        backward = (arcs & 1).astype(bool)
        count = self._prof_off[edges + 1] - self._prof_off[edges]
        first = np.where(backward, self._prof_off[edges + 1] - 1, self._prof_off[edges])
        index = np.repeat(first, count) + np.repeat(np.where(backward, -1, 1), count) * ranges(count)
        spacing = np.repeat(self.length[edges].astype(np.float64) / (count - 1), count)
        # The first sample of every arc but the first repeats the last sample of the arc before it.
        starts = np.cumsum(count) - count
        spacing[starts] = 0.0
        keep = np.ones(len(index), dtype=bool)
        keep[starts[1:]] = False
        return np.cumsum(spacing)[keep], self._prof_ele[index[keep]] / 10.0

    def weights(self, prefs: Prefs) -> np.ndarray:
        return arc_weights(self.g, prefs, self.rides, self.forest, self.ascent)

    def node_niceness(self, w: np.ndarray) -> np.ndarray:
        """Lowest cost multiplier among the arcs leaving each node (1.0 = an ideal road), inf if none.

        Used to place via points on roads the rider would like, not merely on the nearest ones.
        """
        multiplier = w / np.repeat(self.length, 2)
        nice = np.full(self.g.n_nodes, np.inf, dtype=np.float32)
        usable = np.isfinite(multiplier)
        np.minimum.at(nice, self.arc_tail[usable], multiplier[usable])
        return nice
