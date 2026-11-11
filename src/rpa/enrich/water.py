"""Find where the road network touches water: lakes, ponds, reservoirs and rivers from OSM.

The result is a list of water bodies and, for each, the graph nodes close
enough to count as "at the water". Routes "to the water" go through such nodes.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmium
import shapely
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry

from rpa.graph.store import GraphData

# A node this close to the shore counts as being at the water.
ACCESS_DISTANCE_M = 150.0
# Worth riding to: any named lake or pond that is not tiny, a sizeable unnamed
# one, and named rivers. Unnamed river fragments are mostly ditches and culverts.
MIN_AREA_M2 = 50_000.0
MIN_NAMED_AREA_M2 = 5_000.0
# Water nobody wants to ride to.
_UNWANTED_KINDS = {"wastewater", "sewage", "basin", "reflecting_pool", "moat", "drain", "ditch", "pool"}

_KINDS = {"lake", "pond", "reservoir", "river"}

WATER_NODES_FILE = "water_nodes.npy"
WATER_BODIES_FILE = "water.json"


@dataclass
class WaterBody:
    name: str  # empty if unnamed
    kind: str  # "lake", "pond", "reservoir", "river" or plain "water"
    geometry: BaseGeometry  # lon/lat degrees


def read_water(path: Path) -> list[WaterBody]:
    """Water polygons and river lines of an OSM file."""
    factory = osmium.geom.WKBFactory()
    wanted = osmium.filter.TagFilter(("natural", "water"), ("waterway", "river"))
    processor = (
        osmium.FileProcessor(str(path))
        .with_locations()
        .with_areas(wanted)
        .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA | osmium.osm.WAY))
        .with_filter(wanted)
    )
    bodies = []
    for obj in processor:
        tags = obj.tags
        name = tags.get("name", "")
        try:
            if obj.is_area() and tags.get("natural") == "water":
                kind = tags.get("water", "")
                if kind not in _UNWANTED_KINDS:
                    kind = kind if kind in _KINDS else "water"
                    bodies.append(WaterBody(name, kind, shapely.from_wkb(factory.create_multipolygon(obj))))
            elif obj.is_way() and tags.get("waterway") == "river":
                bodies.append(WaterBody(name, "river", shapely.from_wkb(factory.create_linestring(obj))))
        except RuntimeError:
            continue  # broken geometry in OSM (unclosed ring, missing nodes at the extract border)
    return bodies


def water_access(g: GraphData, bodies: list[WaterBody]) -> tuple[np.ndarray, list[dict]]:
    """(u32[K, 2] of (node, body index), descriptions of the bodies that have access nodes).

    Bodies too small to matter are dropped. Body indices refer to the returned list.
    """
    if not bodies:
        return np.empty((0, 2), dtype=np.uint32), []
    lonlat = np.asarray(g.node_lonlat)
    centre_lon = float(lonlat[:, 0].mean())
    utm_zone = int((centre_lon + 180) // 6) + 1
    to_metres = Transformer.from_crs(4326, 32600 + utm_zone, always_xy=True)

    def project(points: np.ndarray) -> np.ndarray:
        return np.stack(to_metres.transform(points[:, 0], points[:, 1]), axis=1)

    shapes = shapely.transform(np.array([body.geometry for body in bodies], dtype=object), project)
    area = shapely.area(shapes)
    named = np.array([bool(body.name) for body in bodies])
    is_river = np.array([body.kind == "river" for body in bodies])
    big_enough = (is_river & named) | (~is_river & ((area >= MIN_AREA_M2) | (named & (area >= MIN_NAMED_AREA_M2))))
    kept = np.flatnonzero(big_enough & shapely.is_valid(shapes))
    if len(kept) == 0:
        return np.empty((0, 2), dtype=np.uint32), []

    tree = shapely.STRtree(shapes[kept])
    points = shapely.points(project(lonlat))
    node, shape = tree.query(points, predicate="dwithin", distance=ACCESS_DISTANCE_M)

    used, body_index = np.unique(shape, return_inverse=True)
    pairs = np.unique(np.stack([node, body_index], axis=1), axis=0).astype(np.uint32)
    described = []
    for i in kept[used]:
        centre = bodies[i].geometry.representative_point()
        described.append(
            {
                "name": bodies[i].name,
                "kind": bodies[i].kind,
                "area_m2": round(float(area[i])),
                "lon": round(centre.x, 6),
                "lat": round(centre.y, 6),
            }
        )
    return pairs, described


def save_water(directory: Path, nodes: np.ndarray, bodies: list[dict]) -> None:
    directory = Path(directory)
    np.save(directory / WATER_NODES_FILE, nodes)
    (directory / WATER_BODIES_FILE).write_text(json.dumps(bodies, ensure_ascii=False))


def load_water(directory: Path) -> tuple[np.ndarray, list[dict]] | None:
    directory = Path(directory)
    if not (directory / WATER_NODES_FILE).exists():
        return None
    return np.load(directory / WATER_NODES_FILE), json.loads((directory / WATER_BODIES_FILE).read_text())
