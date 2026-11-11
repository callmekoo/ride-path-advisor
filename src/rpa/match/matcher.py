"""Map-match cleaned tracks onto OSM ways with Valhalla, running in-process.

The result does not depend on our graph: for each stretch of a ride it records
the OSM way and the two points between which the way was ridden. Projecting
that onto graph edges is `rpa.match.project`, re-run whenever the graph is rebuilt.
"""

import json
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import valhalla

from rpa.graph.store import COORD_SCALE
from rpa.tracks.clean import step_m
from rpa.tracks.store import TrackSet

# Cyclists ride against one-way streets and through "no entry" signs; the
# matcher must follow what was ridden, not what is allowed.
_COSTING = "bicycle"
_COSTING_OPTIONS = {"ignore_oneways": True, "ignore_access": True}
_ATTRIBUTES = ["edge.way_id", "edge.length", "edge.begin_shape_index", "edge.end_shape_index", "shape"]


@dataclass
class Matched:
    way_id: np.ndarray  # i64[M], OSM way of each matched stretch
    ends: np.ndarray  # i32[M, 4], lon1, lat1, lon2, lat2 (* 1e7): where the stretch starts and ends
    track: np.ndarray  # u32[M], index of the track the stretch belongs to
    n_tracks: int


@dataclass
class MatchReport:
    segments: int = 0
    failed: int = 0
    track_km: float = 0.0
    matched_km: float = 0.0


def build_tiles(pbf: Path, directory: Path) -> Path:
    """Build Valhalla routing tiles from `pbf` into `directory`; returns the config path."""
    directory = Path(directory)
    tiles = directory / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    config = valhalla.get_config(tile_dir=tiles, tile_extract="")
    # Defaults reject traces over 200 km or 16k points; long rides exceed both.
    config["service_limits"]["trace"].update(max_distance=2_000_000.0, max_shape=500_000)
    config_path = directory / "valhalla.json"
    config_path.write_text(json.dumps(config, indent=1))
    builder = Path(valhalla.PYVALHALLA_DIR) / "bin" / "valhalla_build_tiles"
    subprocess.run([str(builder), "-c", str(config_path), str(pbf)], check=True, capture_output=True)
    return config_path


def decode_polyline6(text: str) -> np.ndarray:
    """Decode Valhalla's encoded shape into f64[n, 2] lon/lat degrees."""
    values, value, shift = [], 0, 0
    for char in text.encode():
        chunk = char - 63
        value |= (chunk & 0x1F) << shift
        shift += 5
        if chunk < 0x20:
            values.append(~(value >> 1) if value & 1 else value >> 1)
            value, shift = 0, 0
    latlon = np.cumsum(np.array(values, dtype=np.int64).reshape(-1, 2), axis=0) / 1e6
    return latlon[:, ::-1]


_actor: valhalla.Actor | None = None


def _start_worker(config_path: str) -> None:
    global _actor
    _actor = valhalla.Actor(config_path)


def _match_segment(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | str:
    """(way ids, stretch ends, matched km) for one segment, or an error message."""
    request = {
        "shape": [{"lon": float(lon), "lat": float(lat)} for lon, lat in points],
        "costing": _COSTING,
        "shape_match": "map_snap",
        "costing_options": {_COSTING: _COSTING_OPTIONS},
        "filters": {"attributes": _ATTRIBUTES, "action": "include"},
    }
    try:
        response = _actor.trace_attributes(request)
    except Exception as error:  # Valhalla raises on traces it cannot snap
        return str(error)
    if isinstance(response, str):
        response = json.loads(response)
    edges = [edge for edge in response["edges"] if "way_id" in edge]
    shape = np.round(decode_polyline6(response["shape"]) * COORD_SCALE).astype(np.int32)
    begin = np.array([edge["begin_shape_index"] for edge in edges], dtype=np.int64)
    end = np.array([edge["end_shape_index"] for edge in edges], dtype=np.int64)
    way_id = np.array([edge["way_id"] for edge in edges], dtype=np.int64)
    return way_id, np.concatenate([shape[begin], shape[end]], axis=1), sum(edge["length"] for edge in edges)


def match_tracks(
    tracks: TrackSet, config_path: Path, jobs: int, log: Callable[[str], None] = print
) -> tuple[Matched, MatchReport]:
    """Match every segment of `tracks`. Segments Valhalla cannot snap are counted and skipped."""
    report = MatchReport(segments=tracks.n_segments)
    segments = [tracks.segment(s) for s in range(tracks.n_segments)]
    way_ids, ends, owners = [], [], []
    with ProcessPoolExecutor(jobs, initializer=_start_worker, initargs=(str(config_path),)) as pool:
        for s, result in enumerate(pool.map(_match_segment, segments, chunksize=8)):
            report.track_km += float(step_m(segments[s]).sum()) / 1000
            if isinstance(result, str):
                report.failed += 1
                continue
            way_id, stretch_ends, km = result
            report.matched_km += km
            way_ids.append(way_id)
            ends.append(stretch_ends)
            owners.append(np.full(len(way_id), tracks.seg_track[s], dtype=np.uint32))
            if (s + 1) % 500 == 0:
                log(f"  привязано сегментов: {s + 1} из {report.segments}")
    matched = Matched(
        way_id=np.concatenate(way_ids) if way_ids else np.empty(0, dtype=np.int64),
        ends=np.concatenate(ends) if ends else np.empty((0, 4), dtype=np.int32),
        track=np.concatenate(owners) if owners else np.empty(0, dtype=np.uint32),
        n_tracks=len(tracks.tracks),
    )
    return matched, report


def save_matched(matched: Matched, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez(temp, way_id=matched.way_id, ends=matched.ends, track=matched.track, n_tracks=matched.n_tracks)
    temp.replace(path)


def load_matched(path: Path) -> Matched:
    with np.load(path) as data:
        return Matched(data["way_id"], data["ends"], data["track"], int(data["n_tracks"]))
