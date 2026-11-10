"""Turn a sequence of arcs from the kernel into a drawable, describable route."""

from dataclasses import dataclass

import numpy as np

from rpa.graph.store import GraphData
from rpa.osm.tags import Highway, Surface


@dataclass
class RoutePath:
    arcs: np.ndarray  # u32[k], in travel order
    coords: np.ndarray  # f64[P, 2], lon/lat degrees
    length_m: float
    km_by_highway: dict[str, float]  # lower-case Highway name -> km, zero entries omitted
    km_by_surface: dict[str, float]  # lower-case Surface name -> km


def build_path(g: GraphData, arcs: np.ndarray) -> RoutePath:
    """Stitch edge geometries along `arcs` and summarise what the route is made of."""
    arcs = np.asarray(arcs, dtype=np.uint32)
    edges = arcs >> 1
    pieces = []
    for i, (edge, backward) in enumerate(zip(edges.tolist(), (arcs & 1).tolist())):
        piece = g.edge_geometry(edge)
        if backward:
            piece = piece[::-1]
        # Consecutive edges share their joint; keep it once.
        pieces.append(piece if i == 0 else piece[1:])
    coords = np.concatenate(pieces) if pieces else np.empty((0, 2))

    length = np.asarray(g.edge_len_m, dtype=np.float64)[edges]
    return RoutePath(
        arcs=arcs,
        coords=coords,
        length_m=float(length.sum()),
        km_by_highway=_km_by_class(np.asarray(g.edge_highway)[edges], length, Highway),
        km_by_surface=_km_by_class(np.asarray(g.edge_surface)[edges], length, Surface),
    )


def _km_by_class(classes: np.ndarray, length_m: np.ndarray, enum: type) -> dict[str, float]:
    km = np.bincount(classes, weights=length_m, minlength=len(enum)) / 1000
    return {member.name.lower(): float(km[member]) for member in enum if km[member] > 0}
