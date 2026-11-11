"""Spike S1: does in-process Valhalla map-match our tracks, and how fast?

    uv run python spikes/valhalla_match.py [n_segments]
"""

import json
import sys
import time

import numpy as np
from valhalla import Actor

from rpa import config
from rpa.tracks.clean import step_m
from rpa.tracks.store import load_tracks

ATTRIBUTES = ["edge.way_id", "edge.length", "edge.begin_shape_index", "edge.end_shape_index", "shape"]


def match(actor: Actor, points: np.ndarray, costing: str, options: dict) -> dict:
    request = {
        "shape": [{"lon": float(lon), "lat": float(lat)} for lon, lat in points],
        "costing": costing,
        "shape_match": "map_snap",
        "costing_options": {costing: options},
        "filters": {"attributes": ATTRIBUTES, "action": "include"},
    }
    response = actor.trace_attributes(request)
    return json.loads(response) if isinstance(response, str) else response


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tracks = load_tracks(config.tracks_dir())
    actor = Actor(str(config.valhalla_dir() / "valhalla.json"))
    chosen = np.random.default_rng(0).choice(tracks.n_segments, size=count, replace=False)
    for costing, options in [("bicycle", {"ignore_oneways": True, "ignore_access": True}), ("pedestrian", {})]:
        track_km = matched_km = seconds = 0.0
        failures = []
        for s in chosen:
            points = tracks.segment(int(s))
            km = float(step_m(points).sum()) / 1000
            track_km += km
            started = time.perf_counter()
            try:
                result = match(actor, points, costing, options)
                matched_km += sum(edge["length"] for edge in result["edges"])
            except Exception as error:
                failures.append(f"{km:.1f} km: {str(error)[:80]}")
            seconds += time.perf_counter() - started
        print(f"{costing}: matched {matched_km:.1f} of {track_km:.1f} km ({matched_km / track_km:.1%}), "
              f"{seconds / count * 1000:.0f} ms per segment, {len(failures)} failures")
        for failure in failures[:5]:
            print("   ", failure)
    print("sample edge:", result["edges"][0], "| shape length:", len(result["shape"]))


if __name__ == "__main__":
    main()
