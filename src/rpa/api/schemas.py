"""Shapes of what the HTTP API accepts and returns. The request for a route is RouteSpec itself."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from rpa.nlu.prompt import ParsedRequest
from rpa.nlu.resolve import Question
from rpa.routing.spec import Point, RouteSpec

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


class SaveRoute(BaseModel):
    candidate_id: int
    name: PlaceName | None = None  # the candidate's title by default


class SavedRoute(BaseModel):
    id: int
    name: str
    created: str  # UTC, "YYYY-MM-DD HH:MM:SS"
    length_km: float
    ascent_m: float | None


class SavedRouteFull(SavedRoute):
    spec: RouteSpec
    candidate: CandidateBody


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
    has_gazetteer: bool  # place names can be looked up
    has_llm: bool  # free-text requests can be parsed
    can_import_tracks: bool  # ride files can be uploaded and processed


class UploadedFile(BaseModel):
    name: str
    size: int


class TracksStatus(BaseModel):
    state: Literal["idle", "running", "done", "failed"]
    started: str | None  # UTC, "YYYY-MM-DD HH:MM:SS"
    finished: str | None
    log: list[str]  # the last lines that the processing printed
    inbox: list[UploadedFile]  # files waiting to be processed


class ParseRequest(BaseModel):
    text: str = Field(default="", max_length=2000)
    click: Point | None = None  # where the rider put the start marker for this request
    request_id: int | None = None  # the earlier request that this text refines
    picks: dict[str, Point] = {}  # answers to questions, keyed by Question.about
    understood: ParsedRequest | None = None  # resolve this again instead of asking the model


class ParseResponse(BaseModel):
    spec: RouteSpec | None  # ready for POST /api/route
    question: Question | None  # asked instead of a spec when the rider has to settle something
    understood: ParsedRequest
    warnings: list[str]
