"""Shared helpers for tests that work on the micro map (see test_build.py for a sketch)."""

import dataclasses
from pathlib import Path

import numpy as np

from rpa.graph.store import GraphData
from rpa.routing.engine import Engine

MICRO = Path(__file__).parent / "fixtures" / "micro.opl"


def node_index(g: GraphData, osm_id: int) -> int:
    return g.node_osm_id.tolist().index(osm_id)


def edge_index(g: GraphData, osm_u: int, osm_v: int) -> int:
    """Index of the edge stored as osm_u -> osm_v (in OSM way direction)."""
    ids = g.node_osm_id
    matches = np.flatnonzero((ids[g.edge_u] == osm_u) & (ids[g.edge_v] == osm_v))
    assert len(matches) == 1, f"expected exactly one edge {osm_u}->{osm_v}, found {len(matches)}"
    return int(matches[0])


def arc_index(g: GraphData, osm_from: int, osm_to: int) -> int:
    """Arc for travelling osm_from -> osm_to, whichever way the edge is stored."""
    ids = g.node_osm_id
    forward = np.flatnonzero((ids[g.edge_u] == osm_from) & (ids[g.edge_v] == osm_to))
    backward = np.flatnonzero((ids[g.edge_u] == osm_to) & (ids[g.edge_v] == osm_from))
    assert len(forward) + len(backward) == 1, f"ambiguous or missing edge {osm_from}-{osm_to}"
    return 2 * int(forward[0]) if len(forward) else 2 * int(backward[0]) + 1


def grid_graph(nx: int = 21, ny: int = 21, spacing_m: float = 500.0) -> GraphData:
    """A square street grid of plain two-way residential streets.

    Node (i, j) has index j * nx + i; i grows eastwards, j northwards.
    """
    from rpa.osm.tags import BACK_OK, FWD_OK, Highway, Surface

    lon0, lat0 = 60.0, 56.8
    dlat = spacing_m / 111_195.0
    dlon = spacing_m / (111_195.0 * np.cos(np.radians(lat0)))
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    node_lonlat = np.stack([lon0 + i.ravel() * dlon, lat0 + j.ravel() * dlat], axis=1)

    index = (j * nx + i).astype(np.uint32)
    edge_u = np.concatenate([index[:, :-1].ravel(), index[:-1, :].ravel()])
    edge_v = np.concatenate([index[:, 1:].ravel(), index[1:, :].ravel()])
    n_edges = len(edge_u)

    a, b = np.radians(node_lonlat[edge_u]), np.radians(node_lonlat[edge_v])
    h = np.sin((b[:, 1] - a[:, 1]) / 2) ** 2 + np.cos(a[:, 1]) * np.cos(b[:, 1]) * np.sin((b[:, 0] - a[:, 0]) / 2) ** 2
    length = 2 * 6_371_008.8 * np.arcsin(np.sqrt(h))

    geom_xy = np.round(np.stack([node_lonlat[edge_u], node_lonlat[edge_v]], axis=1).reshape(-1, 2) * 1e7)
    graph = GraphData(
        node_lonlat=node_lonlat,
        node_osm_id=np.arange(1, nx * ny + 1, dtype=np.int64),
        edge_u=edge_u,
        edge_v=edge_v,
        edge_len_m=length.astype(np.float32),
        edge_way_id=np.arange(1, n_edges + 1, dtype=np.int64),
        edge_way_pos=np.tile(np.array([0, 1], dtype=np.uint16), (n_edges, 1)),
        edge_highway=np.full(n_edges, Highway.RESIDENTIAL, dtype=np.uint8),
        edge_surface=np.full(n_edges, Surface.UNKNOWN, dtype=np.uint8),
        edge_flags=np.full(n_edges, FWD_OK | BACK_OK, dtype=np.uint16),
        geom_off=(2 * np.arange(n_edges + 1)).astype(np.uint32),
        geom_xy=geom_xy.astype(np.int32),
    )
    graph.validate()
    return graph


def with_edge(g: GraphData, e: int, **attrs) -> GraphData:
    """A copy of the graph where edge `e` gets the given highway/surface/extra flags."""
    changes = {}
    for name, value in attrs.items():
        array = getattr(g, name).copy()
        array[e] = array[e] | value if name == "edge_flags" else value
        changes[name] = array
    return dataclasses.replace(g, **changes)


def with_walked_cut(g: GraphData, i: int, open_rows: range = range(0, 2)) -> GraphData:
    """A copy of the grid where every street crossing between columns i and i + 1 is walk-only.

    The crossings in the rows `open_rows` (the grid's southern edge by default) stay rideable.
    """
    from rpa.osm.tags import DISMOUNT

    rows = np.asarray(g.edge_u) // 21
    cut = (np.asarray(g.edge_u) % 21 == i) & (np.asarray(g.edge_v) % 21 == i + 1) & ~np.isin(rows, list(open_rows))
    flags = g.edge_flags.copy()
    flags[cut] |= DISMOUNT
    return dataclasses.replace(g, edge_flags=flags)


def node(i: int, j: int) -> int:
    return j * 21 + i


def arcs_via(engine: Engine, *nodes: int) -> np.ndarray:
    """Arcs along straight grid lines through the given corner nodes."""
    result = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        step = 1 if abs(b - a) < 21 else 21
        step = step if b > a else -step
        for tail in range(a, b, step):
            match = np.flatnonzero((engine.arc_tail == tail) & (engine.arc_head == tail + step))
            result.append(int(match[0]))
    return np.array(result, dtype=np.uint32)


def square(engine: Engine, i: int, j: int, size: int) -> np.ndarray:
    return arcs_via(
        engine, node(i, j), node(i + size, j), node(i + size, j + size), node(i, j + size), node(i, j)
    )


def without_edges(g: GraphData, drop: np.ndarray) -> GraphData:
    """A copy of the graph without the edges marked in the bool mask `drop`; the nodes stay."""
    keep = np.flatnonzero(~drop)
    pieces = [np.arange(g.geom_off[e], g.geom_off[e + 1]) for e in keep]
    counts = np.array([len(piece) for piece in pieces], dtype=np.uint32)
    graph = dataclasses.replace(
        g,
        edge_u=g.edge_u[keep],
        edge_v=g.edge_v[keep],
        edge_len_m=g.edge_len_m[keep],
        edge_way_id=g.edge_way_id[keep],
        edge_way_pos=g.edge_way_pos[keep],
        edge_highway=g.edge_highway[keep],
        edge_surface=g.edge_surface[keep],
        edge_flags=g.edge_flags[keep],
        geom_off=np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32),
        geom_xy=g.geom_xy[np.concatenate(pieces)],
    )
    graph.validate()
    return graph


def grid_with_hole(g: GraphData, i0: int, i1: int, j0: int, j1: int) -> GraphData:
    """A copy of the 21 x 21 grid with no streets inside the rectangle of nodes i0..i1, j0..j1.

    Every edge at a node strictly inside the rectangle is removed; the streets on its border stay,
    except where the rectangle reaches past the edge of the grid.
    """
    i, j = np.asarray(g.edge_u) % 21, np.asarray(g.edge_u) // 21
    k, m = np.asarray(g.edge_v) % 21, np.asarray(g.edge_v) // 21

    def inside(x, y):
        return (i0 < x) & (x < i1) & (j0 < y) & (y < j1)

    return without_edges(g, inside(i, j) | inside(k, m))


def rect_outline(g: GraphData, i0: int, i1: int, j0: int, j1: int, inset_m: float = 100.0) -> list[tuple[float, float]]:
    """The outline of an object filling the rectangle of nodes i0..i1, j0..j1, `inset_m` inside its border.

    Corners may lie past the edge of the grid; counter-clockwise, as (lon, lat).
    """
    lon0, lat0 = (float(value) for value in g.node_lonlat[0])
    dlat = 500.0 / 111_195.0
    dlon = 500.0 / (111_195.0 * np.cos(np.radians(lat0)))
    west, east = lon0 + i0 * dlon + inset_m / 500 * dlon, lon0 + i1 * dlon - inset_m / 500 * dlon
    south, north = lat0 + j0 * dlat + inset_m / 500 * dlat, lat0 + j1 * dlat - inset_m / 500 * dlat
    return [(west, south), (east, south), (east, north), (west, north)]
