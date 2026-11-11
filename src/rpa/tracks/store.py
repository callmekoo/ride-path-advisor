"""Cleaned tracks on disk: all points in one array, cut into segments, plus per-track metadata."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rpa.graph.store import COORD_SCALE


@dataclass
class TrackSet:
    points: np.ndarray  # i32[P, 2], lon/lat * 1e7
    seg_off: np.ndarray  # u32[S + 1], segment s owns points[seg_off[s]:seg_off[s + 1]]
    seg_track: np.ndarray  # u32[S], index into `tracks`
    tracks: list[dict]  # id, source, date, type, gear, commute

    @property
    def n_segments(self) -> int:
        return len(self.seg_track)

    def segment(self, s: int) -> np.ndarray:
        """Points of segment `s` as f64[k, 2] lon/lat degrees."""
        return self.points[self.seg_off[s] : self.seg_off[s + 1]] / COORD_SCALE


def make_trackset(tracks: list[dict], segments: list[list[np.ndarray]]) -> TrackSet:
    """Pack per-track lists of segments (f64[k, 2] degrees each) into a TrackSet."""
    flat = [segment for per_track in segments for segment in per_track]
    owner = [i for i, per_track in enumerate(segments) for _ in per_track]
    points = np.concatenate(flat) if flat else np.empty((0, 2))
    return TrackSet(
        points=np.round(points * COORD_SCALE).astype(np.int32),
        seg_off=np.cumsum([0] + [len(segment) for segment in flat]).astype(np.uint32),
        seg_track=np.array(owner, dtype=np.uint32),
        tracks=tracks,
    )


def save_tracks(tracks: TrackSet, directory: Path) -> None:
    directory = Path(directory)
    temp = directory.with_name(directory.name + ".tmp")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    np.save(temp / "points.npy", tracks.points)
    np.save(temp / "seg_off.npy", tracks.seg_off)
    np.save(temp / "seg_track.npy", tracks.seg_track)
    (temp / "tracks.json").write_text(json.dumps(tracks.tracks, ensure_ascii=False))
    if directory.exists():
        shutil.rmtree(directory)
    temp.rename(directory)


def load_tracks(directory: Path) -> TrackSet:
    directory = Path(directory)
    return TrackSet(
        points=np.load(directory / "points.npy"),
        seg_off=np.load(directory / "seg_off.npy"),
        seg_track=np.load(directory / "seg_track.npy"),
        tracks=json.loads((directory / "tracks.json").read_text()),
    )
