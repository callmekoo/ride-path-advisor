"""The road graph as a set of flat numpy arrays, and its on-disk form.

A graph lives in a directory of `.npy` files (one per `GraphData` field) plus
`meta.json`. Arrays are memory-mapped on load, so opening a graph is instant.

An edge is a piece of one OSM way between two split nodes; its attributes are
uniform along it. Edge `e` yields two arcs: `2e` runs u -> v, `2e + 1` runs
v -> u.
"""

import dataclasses
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1

# Geometry is stored as integer degrees * 1e7, the native OSM precision.
COORD_SCALE = 1e7


@dataclass
class GraphData:
    node_lonlat: np.ndarray  # f64[N, 2], degrees
    node_osm_id: np.ndarray  # i64[N]
    edge_u: np.ndarray  # u32[E], node at the start of the edge (OSM way direction)
    edge_v: np.ndarray  # u32[E]
    edge_len_m: np.ndarray  # f32[E]
    edge_way_id: np.ndarray  # i64[E]
    edge_way_pos: np.ndarray  # u16[E, 2], first and last node index within the way
    edge_highway: np.ndarray  # u8[E], rpa.osm.tags.Highway
    edge_surface: np.ndarray  # u8[E], rpa.osm.tags.Surface
    edge_flags: np.ndarray  # u16[E], flag bits from rpa.osm.tags
    geom_off: np.ndarray  # u32[E + 1], edge e owns geom_xy[geom_off[e]:geom_off[e + 1]]
    geom_xy: np.ndarray  # i32[P, 2], lon/lat * 1e7 in u -> v order, endpoints included

    @property
    def n_nodes(self) -> int:
        return len(self.node_lonlat)

    @property
    def n_edges(self) -> int:
        return len(self.edge_u)

    def edge_geometry(self, e: int) -> np.ndarray:
        """Polyline of edge `e` as f64[k, 2] lon/lat degrees, in u -> v order."""
        return self.geom_xy[self.geom_off[e] : self.geom_off[e + 1]] / COORD_SCALE

    def validate(self) -> None:
        """Raise ValueError if the arrays do not form a consistent graph."""
        n, e = self.n_nodes, self.n_edges
        expected = {
            "node_lonlat": (np.float64, (n, 2)),
            "node_osm_id": (np.int64, (n,)),
            "edge_u": (np.uint32, (e,)),
            "edge_v": (np.uint32, (e,)),
            "edge_len_m": (np.float32, (e,)),
            "edge_way_id": (np.int64, (e,)),
            "edge_way_pos": (np.uint16, (e, 2)),
            "edge_highway": (np.uint8, (e,)),
            "edge_surface": (np.uint8, (e,)),
            "edge_flags": (np.uint16, (e,)),
            "geom_off": (np.uint32, (e + 1,)),
            "geom_xy": (np.int32, (len(self.geom_xy), 2)),
        }
        for name, (dtype, shape) in expected.items():
            array = getattr(self, name)
            if array.dtype != dtype:
                raise ValueError(f"{name}: dtype is {array.dtype}, expected {np.dtype(dtype)}")
            if array.shape != shape:
                raise ValueError(f"{name}: shape is {array.shape}, expected {shape}")
        if e and max(int(self.edge_u.max()), int(self.edge_v.max())) >= n:
            raise ValueError(f"an edge refers to a node beyond the {n} nodes of the graph")
        if int(self.geom_off[0]) != 0 or int(self.geom_off[-1]) != len(self.geom_xy):
            raise ValueError("geom_off does not cover geom_xy exactly")
        if e and int(np.diff(self.geom_off.astype(np.int64)).min()) < 2:
            raise ValueError("every edge needs at least two geometry points")


_FIELDS = [field.name for field in dataclasses.fields(GraphData)]


def save_graph(g: GraphData, directory: Path, meta: dict) -> None:
    """Write the graph to `directory`, replacing whatever was there.

    Files go to a temporary sibling first, so a reader never sees a
    half-written graph.
    """
    g.validate()
    directory = Path(directory)
    temp = directory.with_name(directory.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    for name in _FIELDS:
        np.save(temp / f"{name}.npy", np.ascontiguousarray(getattr(g, name)))
    full_meta = {**meta, "format_version": FORMAT_VERSION, "n_nodes": g.n_nodes, "n_edges": g.n_edges}
    (temp / "meta.json").write_text(json.dumps(full_meta, indent=2, ensure_ascii=False))
    if directory.exists():
        shutil.rmtree(directory)
    temp.rename(directory)


def load_graph(directory: Path, mmap: bool = True) -> tuple[GraphData, dict]:
    """Load a graph written by `save_graph`. Returns the graph and its metadata."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{directory}: graph format is {meta.get('format_version')}, this code reads "
            f"format {FORMAT_VERSION}; rebuild the graph with `rpa build-graph`"
        )
    mode = "r" if mmap else None
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode=mode) for name in _FIELDS}
    g = GraphData(**arrays)
    g.validate()
    return g, meta
