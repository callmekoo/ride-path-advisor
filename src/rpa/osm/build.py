"""Build the bicycle-routable graph from an OSM file.

Two steps: `read_ways` streams the file once and collects the routable ways
into flat arrays; `assemble` turns those into a `GraphData` with numpy alone,
which keeps it testable without an OSM file.
"""

from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmium
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from rpa.graph.store import COORD_SCALE, GraphData
from rpa.osm.tags import classify_way

EARTH_RADIUS_M = 6_371_008.8

# Mappers sometimes stack two nodes on the same spot. A zero-length edge would
# be free to ride, so give it a token length instead.
MIN_EDGE_LEN_M = 0.1


@dataclass
class RawWays:
    """Routable ways as flat arrays: way `i` owns `refs[way_off[i]:way_off[i + 1]]`."""

    refs: np.ndarray  # i64[R], OSM node ids along the ways
    xy: np.ndarray  # i32[R, 2], lon/lat * 1e7 of those nodes
    way_off: np.ndarray  # u32[W + 1]
    way_id: np.ndarray  # i64[W]
    way_highway: np.ndarray  # u8[W]
    way_surface: np.ndarray  # u8[W]
    way_flags: np.ndarray  # u16[W]


def read_ways(path: Path) -> RawWays:
    """Collect the bicycle-routable ways of an OSM file (.pbf, .osm, .opl)."""
    refs, xs, ys = array("q"), array("i"), array("i")
    way_off = array("I", [0])
    way_id, highway, surface, flags = array("q"), array("B"), array("B"), array("H")

    processor = (
        osmium.FileProcessor(str(path))
        .with_locations()
        .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
        .with_filter(osmium.filter.KeyFilter("highway"))
    )
    for way in processor:
        attrs = classify_way(dict(way.tags))
        if attrs is None:
            continue
        start = len(refs)
        for node in way.nodes:
            location = node.location
            # Ways cut by the border of an extract refer to nodes outside it.
            if location.valid():
                refs.append(node.ref)
                xs.append(location.x)
                ys.append(location.y)
        if len(refs) - start < 2:
            del refs[start:], xs[start:], ys[start:]
            continue
        way_off.append(len(refs))
        way_id.append(way.id)
        highway.append(attrs.highway)
        surface.append(attrs.surface)
        flags.append(attrs.flags)

    return RawWays(
        refs=np.array(refs, dtype=np.int64),
        xy=np.stack([np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)], axis=1),
        way_off=np.array(way_off, dtype=np.uint32),
        way_id=np.array(way_id, dtype=np.int64),
        way_highway=np.array(highway, dtype=np.uint8),
        way_surface=np.array(surface, dtype=np.uint8),
        way_flags=np.array(flags, dtype=np.uint16),
    )


def assemble(raw: RawWays) -> GraphData:
    """Cut ways into edges at split nodes and keep the largest connected part.

    A split node is a way end or a node used more than once among the routable
    ways. An edge is the piece of a way between two consecutive split nodes.
    """
    n_ways = len(raw.way_id)
    if n_ways == 0:
        raise ValueError("no routable ways found")
    refs = raw.refs
    way_off = raw.way_off.astype(np.int64)
    way_of_point = np.repeat(np.arange(n_ways), np.diff(way_off))

    _, inverse, uses = np.unique(refs, return_inverse=True, return_counts=True)
    is_split = uses[inverse] >= 2
    is_split[way_off[:-1]] = True
    is_split[way_off[1:] - 1] = True

    # Consecutive split points within one way bound an edge: points first..last.
    split_pos = np.flatnonzero(is_split)
    first, last = split_pos[:-1], split_pos[1:]
    same_way = way_of_point[first] == way_of_point[last]
    not_loop = refs[first] != refs[last]
    first, last = first[same_way & not_loop], last[same_way & not_loop]
    if len(first) == 0:
        raise ValueError("no routable ways found")

    # Edge length = difference of metres accumulated along the flat point array.
    lonlat = np.radians(raw.xy / COORD_SCALE)
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    h = np.sin(np.diff(lat) / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2
    step_m = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))
    metres = np.concatenate([[0.0], np.cumsum(step_m)])
    length = np.maximum(metres[last] - metres[first], MIN_EDGE_LEN_M)

    end_pos = np.concatenate([first, last])
    node_ids, first_seen = np.unique(refs[end_pos], return_index=True)
    node_xy = raw.xy[end_pos[first_seen]]
    edge_u = np.searchsorted(node_ids, refs[first])
    edge_v = np.searchsorted(node_ids, refs[last])

    # Keep the largest connected component (ignoring one-way directions).
    n_nodes = len(node_ids)
    adjacency = csr_matrix((np.ones(len(edge_u), dtype=np.int8), (edge_u, edge_v)), shape=(n_nodes, n_nodes))
    _, label = connected_components(adjacency, directed=False)
    main = np.bincount(label).argmax()
    keep_node = label == main
    keep_edge = keep_node[edge_u]
    new_index = np.cumsum(keep_node) - 1
    first, last, length = first[keep_edge], last[keep_edge], length[keep_edge]
    edge_u, edge_v = new_index[edge_u[keep_edge]], new_index[edge_v[keep_edge]]

    way = way_of_point[first]
    way_pos = np.stack([first - way_off[way], last - way_off[way]], axis=1)
    if way_pos.max() > np.iinfo(np.uint16).max:
        raise ValueError("a way has more nodes than edge_way_pos can address")

    # Copy each edge's run of points first..last into the geometry array.
    n_points = last - first + 1
    geom_off = np.concatenate([[0], np.cumsum(n_points)])
    point = np.arange(geom_off[-1]) - np.repeat(geom_off[:-1], n_points) + np.repeat(first, n_points)

    return GraphData(
        node_lonlat=node_xy[keep_node] / COORD_SCALE,
        node_osm_id=node_ids[keep_node],
        edge_u=edge_u.astype(np.uint32),
        edge_v=edge_v.astype(np.uint32),
        edge_len_m=length.astype(np.float32),
        edge_way_id=raw.way_id[way],
        edge_way_pos=way_pos.astype(np.uint16),
        edge_highway=raw.way_highway[way],
        edge_surface=raw.way_surface[way],
        edge_flags=raw.way_flags[way],
        geom_off=geom_off.astype(np.uint32),
        geom_xy=raw.xy[point],
    )


def build_graph(path: Path) -> GraphData:
    """Read an OSM file and return its bicycle-routable graph."""
    graph = assemble(read_ways(path))
    graph.validate()
    return graph
