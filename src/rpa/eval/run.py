"""Run eval cases through the generator and check them against quality thresholds."""

import math
import time
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from rpa.eval.cases import Case
from rpa.routing.engine import Engine
from rpa.routing.generate import generate
from rpa.routing.geo import EARTH_RADIUS_M
from rpa.routing.score import Candidate
from rpa.routing.spec import Point

# Pass criteria from the design (spike S3).
MIN_CANDIDATES = 2
MAX_REPEAT_SHARE = 0.05
MAX_SECONDS = 5.0
# A place counts as passed when the route comes this close to it: a place is snapped to a junction nearby.
VIA_REACH_M = 500.0
# A loop without a wanted length comes in three sizes; neighbours must differ at least this much.
MIN_SIZE_STEP = 1.2


@dataclass
class CaseResult:
    case: Case
    candidates: list[Candidate]
    seconds: float
    problems: list[str]  # empty when the case passes


def check(case: Case, candidates: list[Candidate], seconds: float) -> list[str]:
    """What is wrong with the outcome of a case, in words; empty if nothing."""
    problems = []
    if len(candidates) < MIN_CANDIDATES:
        problems.append(f"кандидатов {len(candidates)}, нужно не меньше {MIN_CANDIDATES}")
    length = case.spec.length_km
    for number, candidate in enumerate(candidates, 1):
        km = candidate.path.length_m / 1000
        if length and candidate.note is None and not length.min <= km <= length.max:
            problems.append(f"№{number}: длина {km:.1f} км вне {length.min:.1f}–{length.max:.1f} км")
        if candidate.features.repeat_share > MAX_REPEAT_SHARE:
            problems.append(f"№{number}: повторов {candidate.features.repeat_share:.0%}")
        missed = _missed_place(candidate.path.coords, case.spec.via)
        if missed is not None:
            problems.append(f"№{number}: место «{missed.label or 'без названия'}» не пройдено по порядку")
    if case.spec.shape == "loop" and length is None and not case.spec.via:
        problems += _unlike_sizes([candidate.path.length_m / 1000 for candidate in candidates])
    if case.expect_walk and candidates and candidates[0].features.walk_m == 0:
        problems.append("№1: нет пешего участка")
    if seconds > MAX_SECONDS:
        problems.append(f"считалось {seconds:.1f} с, допустимо {MAX_SECONDS:.0f} с")
    return problems


def _missed_place(coords: np.ndarray, places: list[Point]) -> Point | None:
    """The first of `places` that the line does not come near after having passed the ones before it."""
    at = 0
    for place in places:
        east = np.radians(coords[at:, 0] - place.lon) * math.cos(math.radians(place.lat))
        north = np.radians(coords[at:, 1] - place.lat)
        near = np.flatnonzero(np.hypot(east, north) * EARTH_RADIUS_M <= VIA_REACH_M)
        if len(near) == 0:
            return place
        at += int(near[0])
    return None


def _unlike_sizes(km: list[float]) -> list[str]:
    if len(km) != 3:
        return [f"вариантов {len(km)}, а кольцо без длины должно прийти в три размера"]
    return [
        f"размеры {short:.0f} и {long:.0f} км различаются меньше чем на 20 %"
        for short, long in pairwise(km)
        if long < MIN_SIZE_STEP * short
    ]


def run_case(engine: Engine, case: Case) -> CaseResult:
    sizes = engine.loop_sizes_km
    if case.loop_sizes_km is not None:
        engine.set_loop_sizes(case.loop_sizes_km)
    try:
        started = time.perf_counter()
        candidates = generate(engine, case.spec)
        seconds = time.perf_counter() - started
    finally:
        engine.set_loop_sizes(sizes)
    return CaseResult(case, candidates, seconds, check(case, candidates, seconds))
