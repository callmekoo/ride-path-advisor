"""Shapes of what the HTTP API accepts and returns. The request for a route is RouteSpec itself."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from rpa.routing.spec import RouteSpec

PlaceName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Lon = Annotated[float, Field(ge=-180, le=180)]
Lat = Annotated[float, Field(ge=-90, le=90)]


class PlaceIn(BaseModel):
    name: PlaceName
    lon: Lon
    lat: Lat
    is_default: bool = False


class PlaceChange(BaseModel):
    """Only the given fields change."""

    name: PlaceName | None = None
    lon: Lon | None = None
    lat: Lat | None = None
    is_default: bool | None = None


class PlaceOut(BaseModel):
    id: int
    name: str
    lon: float
    lat: float
    is_default: bool


class LineString(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: list[tuple[float, float]]  # lon, lat


class Profile(BaseModel):
    distance_m: list[float]
    elevation_m: list[float]


class CandidateBody(BaseModel):
    """A route as stored and shown; everything but its id, which the database assigns."""

    title: str
    geometry: LineString
    length_km: float
    ascent_m: float | None  # None when the graph has no elevation
    score: float
    note: str | None
    via_label: str | None
    features: dict[str, float]
    km_by_highway: dict[str, float]
    km_by_surface: dict[str, float]
    profile: Profile | None


class CandidateOut(CandidateBody):
    id: int


class RouteResponse(BaseModel):
    request_id: int
    spec: RouteSpec
    candidates: list[CandidateOut]
    warnings: list[str]
    message: str | None  # why there are no candidates


class Meta(BaseModel):
    nodes: int
    edges: int
    road_km: float
    bbox: tuple[float, float, float, float]  # west, south, east, north
    has_rides: bool
    ridden_km: float
    has_water: bool
    has_terrain: bool
