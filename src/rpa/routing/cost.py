"""Turn riding preferences into one weight per arc.

    w = length * (1 + m * road_class_penalty + f * unfamiliarity) * surface_factor * restriction_factor

Every factor is at least 1, so a finite weight never undercuts the edge
length. The kernel's A* heuristic depends on that. Forbidden arcs get +inf.

The numbers are starting points, to be tuned on the eval page.
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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
# Having actually ridden a restricted way this many times shows it is passable.
RIDES_LIFTING_RESTRICTION = 2
# Extra metres per metre on a never-ridden road at familiarity = 1.
UNFAMILIAR_PENALTY = 1.5
# This many rides make a road fully familiar; fewer count on a log scale.
FULLY_FAMILIAR_RIDES = 30
FORD_FACTOR = 1.5
MAX_MULTIPLIER = 50.0

_PENALTY_BY_CLASS = np.array([_ROAD_CLASS_PENALTY[h] for h in Highway])
assert all(len(factors) == len(Surface) for factors in _SURFACE_FACTOR.values())


class Prefs(BaseModel):
    """Riding preferences that shape the arc weights."""

    model_config = ConfigDict(frozen=True)

    avoid_major: float = Field(default=0.5, ge=0.0, le=1.0)  # 0 = big roads are fine, 1 = avoid them hard
    surface: Literal["any", "paved", "prefer_unpaved"] = "any"
    familiarity: float = Field(default=0.4, ge=0.0, le=1.0)  # 0 = ignore ride history, 1 = stick to ridden roads


def popularity(rides: np.ndarray) -> np.ndarray:
    """Ride counts on a 0..1 scale: 0 = never ridden, 1 = ridden FULLY_FAMILIAR_RIDES times or more."""
    return np.minimum(1.0, np.log1p(rides) / np.log1p(FULLY_FAMILIAR_RIDES))


def arc_weights(g: GraphData, prefs: Prefs, rides: np.ndarray | None = None) -> np.ndarray:
    """Weights for the kernel: f32[2E], arc 2e runs u -> v and arc 2e + 1 runs v -> u.

    `rides` (per edge, how many recorded rides used it) enables the familiarity
    preference; without it every road counts as equally unknown.
    """
    flags = np.asarray(g.edge_flags)
    major_scale = 0.3 + 1.7 * prefs.avoid_major
    class_penalty = _PENALTY_BY_CLASS[g.edge_highway]
    class_penalty = np.where(flags & CYCLE_INFRA, class_penalty / 2, class_penalty)

    multiplier = 1.0 + major_scale * class_penalty
    restricted = (flags & SOFT_RESTRICTED) > 0
    if rides is not None:
        multiplier = multiplier + UNFAMILIAR_PENALTY * prefs.familiarity * (1.0 - popularity(rides))
        restricted &= rides < RIDES_LIFTING_RESTRICTION
    multiplier = multiplier * np.array(_SURFACE_FACTOR[prefs.surface])[g.edge_surface]
    multiplier = multiplier * np.where(restricted, SOFT_RESTRICTION_FACTOR, 1.0)
    multiplier = multiplier * np.where(flags & FORD, FORD_FACTOR, 1.0)
    # The cap keeps penalties from compounding into a de-facto ban; real bans stay infinite.
    multiplier = np.where(np.isfinite(multiplier), np.minimum(multiplier, MAX_MULTIPLIER), _INF)

    edge_weight = np.asarray(g.edge_len_m, dtype=np.float64) * multiplier
    weights = np.empty(2 * g.n_edges, dtype=np.float32)
    weights[0::2] = np.where(flags & FWD_OK, edge_weight, _INF)
    weights[1::2] = np.where(flags & BACK_OK, edge_weight, _INF)
    return weights
