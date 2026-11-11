"""Sample public rasters along the edges of the graph: elevation and tree cover.

Elevation comes from Copernicus DEM GLO-30, tree cover from ESA WorldCover.
Both are read straight from their public buckets, tile by tile; nothing but the
per-edge results is kept.
"""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from affine import Affine
from scipy import ndimage

from rpa.arrays import ranges
from rpa.graph.store import COORD_SCALE, GraphData
from rpa.tracks.clean import step_m

SAMPLE_SPACING_M = 50.0
# WorldCover is read at 1/8 of its 10 m resolution: plenty for "is this road in a forest".
_WORLDCOVER_OVERVIEW = 2
_WORLDCOVER_TREES = 10
_DEM_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{name}/{name}.tif"
_WORLDCOVER_URL = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{name}_Map.tif"
)

# (array, transform) of a tile, or None if the tile does not exist (open sea).
Tile = tuple[np.ndarray, Affine] | None
TileReader = Callable[[int, int], Tile]


def edge_samples(g: GraphData, spacing_m: float = SAMPLE_SPACING_M) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced points along every edge, both ends included.

    Returns (lonlat f64[S, 2], off u32[E + 1]); edge e owns samples off[e]:off[e + 1],
    in u -> v order, at least two of them.
    """
    geom = np.asarray(g.geom_xy) / COORD_SCALE
    geom_off = np.asarray(g.geom_off).astype(np.int64)
    step = np.concatenate([[0.0], step_m(geom)])
    step[geom_off[:-1]] = 0.0
    along = np.cumsum(step)
    length = along[geom_off[1:] - 1] - along[geom_off[:-1]]
    # Shift every edge by one more metre, so positions strictly increase from edge to edge.
    position = along + np.repeat(np.arange(g.n_edges, dtype=np.float64), np.diff(geom_off))

    count = np.maximum(2, np.ceil(length / spacing_m).astype(np.int64) + 1)
    share = ranges(count) / np.repeat(count - 1, count)
    wanted = np.repeat(position[geom_off[:-1]], count) + share * np.repeat(length, count)
    lonlat = np.stack([np.interp(wanted, position, geom[:, 0]), np.interp(wanted, position, geom[:, 1])], axis=1)
    return lonlat, np.concatenate([[0], np.cumsum(count)]).astype(np.uint32)


def sample_tiles(
    lonlat: np.ndarray, tile_deg: int, read_tile: TileReader, bilinear: bool, log: Callable[[str], None] = print
) -> np.ndarray:
    """Raster value at each point: f32[S]. Tiles are `tile_deg` degrees wide, named by their south-west corner."""
    lat0 = (np.floor(lonlat[:, 1] / tile_deg) * tile_deg).astype(np.int64)
    lon0 = (np.floor(lonlat[:, 0] / tile_deg) * tile_deg).astype(np.int64)
    keys, which = np.unique(np.stack([lat0, lon0], axis=1), axis=0, return_inverse=True)
    values = np.zeros(len(lonlat), dtype=np.float32)

    with ThreadPoolExecutor(6) as pool:
        tiles = pool.map(lambda key: read_tile(int(key[0]), int(key[1])), keys)
        for number, tile in enumerate(tiles):
            if (number + 1) % 20 == 0:
                log(f"  плиток: {number + 1} из {len(keys)}")
            if tile is None:
                continue
            array, transform = tile
            index = np.flatnonzero(which.ravel() == number)
            col = (lonlat[index, 0] - transform.c) / transform.a
            row = (lonlat[index, 1] - transform.f) / transform.e
            if bilinear:
                values[index] = ndimage.map_coordinates(array, [row - 0.5, col - 0.5], order=1, mode="nearest")
            else:
                rows = np.clip(np.floor(row).astype(np.int64), 0, array.shape[0] - 1)
                cols = np.clip(np.floor(col).astype(np.int64), 0, array.shape[1] - 1)
                values[index] = array[rows, cols]
    return values


def _tile_name(lat0: int, lon0: int, lat_digits: int = 2) -> tuple[str, str]:
    return (f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):0{lat_digits}d}", f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}")


def _read_remote(url: str, **options) -> Tile:
    """Read band 1 of a remote GeoTIFF; None if it does not exist. Retries on network trouble."""
    for attempt in range(4):
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                with rasterio.open(url, **options) as source:
                    return source.read(1), source.transform
        except rasterio.errors.RasterioIOError as error:
            if "404" in str(error) or "does not exist" in str(error):
                return None
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    return None


def read_dem_tile(lat0: int, lon0: int) -> Tile:
    """Elevation in metres for the 1-degree Copernicus GLO-30 tile with this south-west corner."""
    lat, lon = _tile_name(lat0, lon0)
    return _read_remote(_DEM_URL.format(name=f"Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM"))


def read_tree_tile(lat0: int, lon0: int) -> Tile:
    """Share of tree cover (0..1) around each pixel of the 3-degree WorldCover tile."""
    lat, lon = _tile_name(lat0, lon0)
    tile = _read_remote(_WORLDCOVER_URL.format(name=lat + lon), overview_level=_WORLDCOVER_OVERVIEW)
    if tile is None:
        return None
    classes, transform = tile
    # Averaging over the neighbouring pixels tells "in a forest" from "past a lone copse".
    return ndimage.uniform_filter((classes == _WORLDCOVER_TREES).astype(np.float32), size=3), transform


def forest_share(lonlat: np.ndarray, off: np.ndarray, read_tile: TileReader = read_tree_tile, log=print) -> np.ndarray:
    """u8[E]: share of each edge that runs through forest, scaled to 0..255."""
    trees = sample_tiles(lonlat, 3, read_tile, bilinear=False, log=log)
    count = np.diff(off.astype(np.int64))
    mean = np.add.reduceat(trees.astype(np.float64), off[:-1].astype(np.int64)) / count
    return np.round(np.clip(mean, 0.0, 1.0) * 255).astype(np.uint8)


def elevation(lonlat: np.ndarray, read_tile: TileReader = read_dem_tile, log=print) -> np.ndarray:
    """i16[S]: elevation of each sample in decimetres."""
    metres = sample_tiles(lonlat, 1, read_tile, bilinear=True, log=log)
    return np.round(np.clip(metres * 10, -32000, 32000)).astype(np.int16)


def edge_ascent(off: np.ndarray, ele_dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ascent riding u -> v, ascent riding v -> u), f32[E] metres each.

    The surface model is noisy (it sees treetops), so each edge's profile is
    smoothed over five samples before its ups and downs are added up. This is
    for costing; a route's total climb is computed from its whole profile.
    """
    off = off.astype(np.int64)
    count = np.diff(off)
    n_edges = len(count)
    ele = ele_dm.astype(np.float64) / 10
    index = np.arange(len(ele))
    edge = np.repeat(np.arange(n_edges), count)
    low = np.maximum(index - 2, off[edge])
    high = np.minimum(index + 3, off[edge + 1])
    prefix = np.concatenate([[0.0], np.cumsum(ele)])
    smooth = (prefix[high] - prefix[low]) / (high - low)

    change = np.diff(smooth)
    inside = edge[1:] == edge[:-1]
    ups = np.bincount(edge[1:][inside], weights=np.maximum(change[inside], 0.0), minlength=n_edges)
    downs = np.bincount(edge[1:][inside], weights=np.maximum(-change[inside], 0.0), minlength=n_edges)
    # Smoothing flattens the ends of an edge, so take the overall rise from the
    # raw end points and from the smoothed profile only the ups and downs beyond it.
    net = ele[off[1:] - 1] - ele[off[:-1]]
    net_smooth = smooth[off[1:] - 1] - smooth[off[:-1]]
    up = np.maximum(net, 0.0) + (ups - np.maximum(net_smooth, 0.0))
    down = np.maximum(-net, 0.0) + (downs - np.maximum(-net_smooth, 0.0))
    return up.astype(np.float32), down.astype(np.float32)


def climb_m(ele_m: np.ndarray, window: int = 5, threshold_m: float = 5.0) -> float:
    """Total ascent of a profile: smoothed, and counting only rises of `threshold_m` or more."""
    if len(ele_m) < 2:
        return 0.0
    pad = window // 2
    smooth = np.convolve(np.pad(ele_m, pad, mode="edge"), np.ones(window) / window, mode="valid")
    total = 0.0
    valley = peak = float(smooth[0])
    for value in smooth[1:].tolist():
        if value > peak:
            peak = value
        elif peak - value >= threshold_m:
            # A real descent: the climb from the valley to the peak is over.
            if peak - valley >= threshold_m:
                total += peak - valley
            valley = peak = value
        elif value < valley:
            valley = peak = value
    if peak - valley >= threshold_m:
        total += peak - valley
    return total


def tile_count(lonlat: np.ndarray, tile_deg: int) -> int:
    keys = np.stack([np.floor(lonlat[:, 1] / tile_deg), np.floor(lonlat[:, 0] / tile_deg)], axis=1)
    return len(np.unique(keys, axis=0))

