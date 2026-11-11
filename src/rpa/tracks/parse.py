"""Read GPS points out of GPX, TCX and FIT files."""

import gzip
import io
from dataclasses import dataclass
from datetime import datetime
from xml.etree import ElementTree

import fitdecode
import numpy as np

_SEMICIRCLE_DEG = 180.0 / 2**31


@dataclass
class RawTrack:
    lonlat: np.ndarray  # f64[P, 2], degrees, as recorded
    time: np.ndarray  # f64[P], unix seconds; NaN where the file has no time


def kind_of(filename: str) -> tuple[str, bool] | None:
    """("gpx" | "tcx" | "fit", is_gzipped) for a track file name, or None for anything else."""
    name = filename.lower()
    gzipped = name.endswith(".gz")
    if gzipped:
        name = name[:-3]
    kind = name.rsplit(".", 1)[-1]
    return (kind, gzipped) if kind in ("gpx", "tcx", "fit") else None


def parse_track(data: bytes, kind: str, gzipped: bool = False) -> RawTrack:
    if gzipped:
        data = gzip.decompress(data)
    points = {"gpx": _gpx_points, "tcx": _tcx_points, "fit": _fit_points}[kind](data)
    if not points:
        return RawTrack(np.empty((0, 2)), np.empty(0))
    array = np.array(points, dtype=np.float64)
    return RawTrack(array[:, :2], array[:, 2])


def _unix(text: str | None) -> float:
    if not text:
        return float("nan")
    try:
        return datetime.fromisoformat(text.strip()).timestamp()
    except ValueError:
        return float("nan")


def _gpx_points(data: bytes) -> list[tuple[float, float, float]]:
    points = []
    for point in ElementTree.fromstring(data).iterfind(".//{*}trkpt"):
        points.append((float(point.get("lon")), float(point.get("lat")), _unix(point.findtext("{*}time"))))
    return points


def _tcx_points(data: bytes) -> list[tuple[float, float, float]]:
    # Strava's TCX files start with stray whitespace, which XML parsers reject.
    points = []
    for point in ElementTree.fromstring(data.lstrip()).iterfind(".//{*}Trackpoint"):
        lat = point.findtext("{*}Position/{*}LatitudeDegrees")
        lon = point.findtext("{*}Position/{*}LongitudeDegrees")
        if lat is not None and lon is not None:
            points.append((float(lon), float(lat), _unix(point.findtext("{*}Time"))))
    return points


def _fit_points(data: bytes) -> list[tuple[float, float, float]]:
    points = []
    with fitdecode.FitReader(io.BytesIO(data)) as reader:
        for frame in reader:
            if not isinstance(frame, fitdecode.FitDataMessage) or frame.name != "record":
                continue
            lat = frame.get_value("position_lat", fallback=None)
            lon = frame.get_value("position_long", fallback=None)
            if lat is None or lon is None:
                continue
            stamp = frame.get_value("timestamp", fallback=None)
            seconds = stamp.timestamp() if isinstance(stamp, datetime) else float("nan")
            points.append((lon * _SEMICIRCLE_DEG, lat * _SEMICIRCLE_DEG, seconds))
    return points
