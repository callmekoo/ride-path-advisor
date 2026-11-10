"""Turn riding preferences into one weight per arc.

    w = length * (1 + m * road_class_penalty) * surface_factor * restriction_factor

Every factor is at least 1, so a finite weight never undercuts the edge
length. The kernel's A* heuristic depends on that. Forbidden arcs get +inf.

The numbers are starting points, to be tuned on the eval page.
"""

from dataclasses import dataclass

import numpy as np

from rpa.graph.store import GraphData
from rpa.osm.tags import BACK_OK, CYCLE_INFRA, FORD, FWD_OK, SOFT_RESTRICTED, Highway, Surface

# Extra metres per metre for riding on a road of this class, before scaling
# by the rider's wish to avoid major roads.
_ROAD_CLASS_PENALTY = {
    Highway.CYCLEWAY: 0.0,
    Highway.UNCLASSIFIED: 0.0,
    Highway.TRACK: 0.05,
    Highway.RESIDENTIAL: 0.1,
    Highway.LIVING_STREET: 0.1,
    Highway.SERVICE: 0.15,
    Highway.TERTIARY: 0.15,
    Highway.PATH: 0.15,
    Highway.PEDESTRIAN: 0.2,
    Highway.FOOTWAY: 0.3,
    Highway.SECONDARY: 0.5,
    Highway.PRIMARY: 1.2,
    Highway.TRUNK: 3.0,
}

_INF = float("inf")

# Factor per Surface class (paved .. bad) for each surface mode.
_SURFACE_FACTOR = {
    "any": (1.0, 1.0, 1.05, 1.1, 1.15, 1.8),
    "paved": (1.0, 1.1, 2.0, 5.0, 12.0, _INF),
    "prefer_unpaved": (1.6, 1.4, 1.15, 1.0, 1.0, 1.5),
}

SOFT_RESTRICTION_FACTOR = 3.0
FORD_FACTOR = 1.5
MAX_MULTIPLIER = 50.0

_PENALTY_BY_CLASS = np.array([_ROAD_CLASS_PENALTY[h] for h in Highway])
assert all(len(factors) == len(Surface) for factors in _SURFACE_FACTOR.values())


@dataclass(frozen=True)
class Prefs:
    avoid_major: float = 0.5  # 0 = big roads are fine, 1 = avoid them hard
    surface: str = "any"  # "any" | "paved" | "prefer_unpaved"

    def __post_init__(self) -> None:
        if not 0.0 <= self.avoid_major <= 1.0:
            raise ValueError(f"avoid_major must be within 0..1, got {self.avoid_major}")
        if self.surface not in _SURFACE_FACTOR:
            raise ValueError(f"surface must be one of {sorted(_SURFACE_FACTOR)}, got {self.surface!r}")


def arc_weights(g: GraphData, prefs: Prefs) -> np.ndarray:
    """Weights for the kernel: f32[2E], arc 2e runs u -> v and arc 2e + 1 runs v -> u."""
    flags = np.asarray(g.edge_flags)
    major_scale = 0.3 + 1.7 * prefs.avoid_major
    class_penalty = _PENALTY_BY_CLASS[g.edge_highway]
    class_penalty = np.where(flags & CYCLE_INFRA, class_penalty / 2, class_penalty)

    multiplier = 1.0 + major_scale * class_penalty
    multiplier = multiplier * np.array(_SURFACE_FACTOR[prefs.surface])[g.edge_surface]
    multiplier = multiplier * np.where(flags & SOFT_RESTRICTED, SOFT_RESTRICTION_FACTOR, 1.0)
    multiplier = multiplier * np.where(flags & FORD, FORD_FACTOR, 1.0)
    # The cap keeps penalties from compounding into a de-facto ban; real bans stay infinite.
    multiplier = np.where(np.isfinite(multiplier), np.minimum(multiplier, MAX_MULTIPLIER), _INF)

    edge_weight = np.asarray(g.edge_len_m, dtype=np.float64) * multiplier
    weights = np.empty(2 * g.n_edges, dtype=np.float32)
    weights[0::2] = np.where(flags & FWD_OK, edge_weight, _INF)
    weights[1::2] = np.where(flags & BACK_OK, edge_weight, _INF)
    return weights
