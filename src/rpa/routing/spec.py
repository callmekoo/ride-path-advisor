"""RouteSpec: the structured request that the route engine consumes.

The text parser and the UI both produce a RouteSpec; the engine reads nothing
else. That keeps the two sides independently testable.
"""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from rpa.routing.cost import Prefs


class Point(BaseModel):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    label: str | None = None


class LengthKm(BaseModel):
    """Wanted route length: aim for `target`, accept anything within `min..max`."""

    target: float
    min: float = Field(gt=0)
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.min <= self.target <= self.max:
            raise ValueError(f"length must satisfy min <= target <= max, got {self.min}, {self.target}, {self.max}")
        return self

    @classmethod
    def around(cls, km: float, tolerance: float = 0.1) -> Self:
        """ "About 45 km": the target with some slack either way."""
        return cls(target=km, min=km * (1 - tolerance), max=km * (1 + tolerance))

    @classmethod
    def between(cls, low: float, high: float) -> Self:
        """ "100 to 150 km": a hard range, aiming for the middle."""
        return cls(target=(low + high) / 2, min=low, max=high)


class AvoidArea(BaseModel):
    """A circle the route should keep out of if it reasonably can."""

    point: Point
    radius_m: float = Field(ge=100, le=50_000)


class RouteSpec(BaseModel):
    start: Point
    end: Point | None = None
    shape: Literal["loop", "p2p"]
    length_km: LengthKm | None = None
    prefs: Prefs = Prefs()
    via: Point | None = None  # a place the route must pass; its label names it in the result
    via_category: Literal["water"] | None = None  # the route must pass a place of this kind
    avoid: list[AvoidArea] = []
    bearing_deg: float | None = Field(default=None, ge=0, lt=360)  # where a loop should head; 0 = north
    n_candidates: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def _shape_matches_fields(self) -> Self:
        if self.via is not None and self.via_category is not None:
            raise ValueError("a route passes either a named place or a kind of place, not both")
        if self.shape == "loop":
            if self.end is not None:
                raise ValueError("a loop returns to its start and takes no end point")
            if self.length_km is None:
                raise ValueError("a loop needs a length")
        elif self.end is None:
            raise ValueError("a point-to-point route needs an end point")
        return self
