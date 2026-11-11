"""Run eval cases through the generator and check them against quality thresholds."""

import time
from dataclasses import dataclass

from rpa.eval.cases import Case
from rpa.routing.engine import Engine
from rpa.routing.generate import generate
from rpa.routing.score import Candidate

# Pass criteria from the design (spike S3).
MIN_CANDIDATES = 2
MAX_REPEAT_SHARE = 0.05
MAX_SECONDS = 5.0


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
    if seconds > MAX_SECONDS:
        problems.append(f"считалось {seconds:.1f} с, допустимо {MAX_SECONDS:.0f} с")
    return problems


def run_case(engine: Engine, case: Case) -> CaseResult:
    started = time.perf_counter()
    candidates = generate(engine, case.spec)
    seconds = time.perf_counter() - started
    return CaseResult(case, candidates, seconds, check(case, candidates, seconds))
