"""The HTTP API: routes for a RouteSpec, GPX export, saved places and routes; and the web page, if built.

Handlers are plain functions, so FastAPI runs them in its thread pool; the search
kernel releases the GIL, which lets requests overlap.
"""

import json
import logging
import threading
from dataclasses import asdict
from functools import cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from rpa.api.db import Db, DuplicateName
from rpa.api.jobs import Refused, TrackImport
from rpa.api.schemas import (
    CandidateBody,
    CandidateOut,
    LineString,
    Meta,
    ParseRequest,
    ParseResponse,
    PlaceChange,
    PlaceIn,
    PlaceOut,
    Profile,
    RouteResponse,
    SavedRoute,
    SavedRouteFull,
    SaveRoute,
    TracksStatus,
    UploadedFile,
)
from rpa.nlu.client import LlmClient, LlmError
from rpa.nlu.prompt import ParsedRequest, describe, messages
from rpa.nlu.resolve import Context, resolve
from rpa.osm.gazetteer import Gazetteer
from rpa.routing.engine import Engine
from rpa.routing.generate import generate
from rpa.routing.gpx import to_gpx
from rpa.routing.score import Candidate
from rpa.routing.spec import Point, RouteSpec
from rpa.viz.heat_page import ridden_roads

# Beyond FAR_SNAP_M a snapped point probably is not where the user meant; beyond
# MAX_SNAP_M it is off the map and a route from the nearest road would mislead.
FAR_SNAP_M = 500.0
MAX_SNAP_M = 2000.0
MAX_PROFILE_POINTS = 500

_NO_WATER_DATA = "В графе нет данных о воде: выполните rpa enrich."
_NOTHING_FOUND = "Не удалось построить маршрут с такими условиями. Попробуйте другую длину или более мягкие ограничения."


_NO_RIDES = "Истории заездов нет: импортируйте треки (rpa import-tracks, rpa match)."
_NO_SUCH_ROUTE = "Такого сохранённого маршрута нет."
_NO_UPLOADS = "Загрузка треков на этом сервере не настроена."
_NO_LLM = "Разбор текста не настроен: задайте LLM_BASE_URL, LLM_MODEL и LLM_API_KEY (см. .env.example)."


def create_app(
    engine: Engine,
    db_path: Path,
    gazetteer: Gazetteer | None = None,
    llm: LlmClient | None = None,
    static_dir: Path | None = None,
    rides_path: Path | None = None,
    tracks: TrackImport | None = None,
) -> FastAPI:
    """The application.

    `static_dir` is the built web page, served from the root when it is there. `rides_path` is the
    ride counts file next to the graph: when it is replaced, the counts are read again. `tracks`
    takes uploaded ride files and processes them.
    """
    app = FastAPI(title="ride-path-advisor")
    app.add_middleware(GZipMiddleware, minimum_size=2000)
    db = Db(db_path)
    rides_file = _RidesFile(rides_path, engine) if rides_path is not None else None

    @cache
    def about() -> Meta:
        capabilities = {"has_gazetteer": gazetteer is not None, "has_llm": llm is not None}
        return _meta(engine).model_copy(update=capabilities | {"can_import_tracks": tracks is not None})

    @cache
    def heatmap_json() -> bytes:
        # A few megabytes for a rider's whole history: built on first use, then kept.
        return json.dumps(ridden_roads(engine.g, engine.rides), separators=(",", ":")).encode()

    def with_current_rides() -> None:
        """Whatever depends on the ride history starts with this: new rides may have been matched meanwhile."""
        if rides_file is not None and rides_file.refresh():
            about.cache_clear()
            heatmap_json.cache_clear()

    @app.get("/api/meta")
    def get_meta() -> Meta:
        with_current_rides()
        return about()

    @app.post("/api/route")
    def post_route(spec: RouteSpec) -> RouteResponse:
        with_current_rides()
        warnings = _check_point(engine, spec.start, "старт")
        if spec.end is not None:
            warnings += _check_point(engine, spec.end, "финиш")

        bodies = [_body(candidate, spec) for candidate in generate(engine, spec)]
        message = None
        if not bodies:
            message = _NO_WATER_DATA if spec.via_category == "water" and engine.at_water is None else _NOTHING_FOUND
        request_id, ids = db.save_request(spec.model_dump_json(), [body.model_dump_json() for body in bodies])
        return RouteResponse(
            request_id=request_id,
            spec=spec,
            candidates=[CandidateOut(id=i, **body.model_dump()) for i, body in zip(ids, bodies)],
            warnings=warnings,
            message=message,
        )

    @app.post("/api/parse")
    def post_parse(body: ParseRequest) -> ParseResponse:
        previous, offered = None, []
        if body.request_id is not None:
            stored = db.request(body.request_id)
            if stored is None:
                raise HTTPException(404, "Предыдущий запрос не найден: возможно, он устарел. Начните заново.")
            previous = RouteSpec.model_validate_json(stored["spec"])
            offered = [CandidateBody.model_validate_json(candidate) for candidate in stored["candidates"]]
        saved_places = db.places()

        parsed = body.understood
        if parsed is None:
            if not body.text.strip():
                raise HTTPException(422, "Пустой запрос.")
            if llm is None:
                raise HTTPException(503, _NO_LLM)
            lines = [_candidate_line(number, candidate) for number, candidate in enumerate(offered, start=1)]
            names = [place["name"] for place in saved_places]
            try:
                # Only words go to the model: the text, place names and candidate statistics.
                parsed = llm.parse(
                    *messages(body.text, describe(previous) if previous else None, lines, names), ParsedRequest
                )
            except LlmError as error:
                raise HTTPException(502, str(error)) from None

        context = Context(
            snapper=engine.snapper,
            gazetteer=gazetteer,
            saved_places=saved_places,
            click=body.click,
            previous=previous,
            previous_candidates=[candidate.geometry.coordinates for candidate in offered],
            picks=body.picks,
        )
        resolved = resolve(parsed, context)
        return ParseResponse(
            spec=resolved.spec, question=resolved.question, understood=parsed, warnings=resolved.warnings
        )

    @app.get("/api/candidates/{candidate_id}.gpx")
    def get_gpx(candidate_id: int) -> Response:
        stored = db.candidate(candidate_id)
        if stored is None:
            raise HTTPException(404, "Маршрут не найден: возможно, он устарел. Постройте его заново.")
        return _gpx(CandidateBody.model_validate_json(stored), None, f"route-{candidate_id}")

    @app.post("/api/routes", status_code=201)
    def post_saved_route(wanted: SaveRoute) -> SavedRoute:
        saved = db.save_route(wanted.candidate_id, wanted.name)
        if saved is None:
            raise HTTPException(404, "Маршрут не найден: возможно, он устарел. Постройте его заново.")
        return saved

    @app.get("/api/routes")
    def get_saved_routes() -> list[SavedRoute]:
        return db.routes()

    @app.get("/api/routes/{route_id}.gpx")  # before the plain id, which would swallow "5.gpx"
    def get_saved_gpx(route_id: int) -> Response:
        saved = db.route(route_id)
        if saved is None:
            raise HTTPException(404, _NO_SUCH_ROUTE)
        return _gpx(CandidateBody.model_validate_json(saved["body"]), saved["name"], f"saved-{route_id}")

    @app.get("/api/routes/{route_id}")
    def get_saved_route(route_id: int) -> SavedRouteFull:
        saved = db.route(route_id)
        if saved is None:
            raise HTTPException(404, _NO_SUCH_ROUTE)
        return SavedRouteFull(
            **{key: saved[key] for key in ("id", "name", "created", "length_km", "ascent_m")},
            spec=RouteSpec.model_validate_json(saved["spec"]),
            candidate=CandidateBody.model_validate_json(saved["body"]),
        )

    @app.delete("/api/routes/{route_id}", status_code=204)
    def delete_saved_route(route_id: int) -> None:
        if not db.delete_route(route_id):
            raise HTTPException(404, _NO_SUCH_ROUTE)

    @app.get("/api/heatmap")
    def get_heatmap() -> Response:
        """Roads from the ride history as GeoJSON: one MultiLineString per level of use."""
        with_current_rides()
        if engine.rides is None:
            raise HTTPException(404, _NO_RIDES)
        return Response(heatmap_json(), media_type="application/geo+json")

    @app.get("/api/places")
    def get_places() -> list[PlaceOut]:
        return db.places()

    @app.post("/api/places", status_code=201)
    def post_place(place: PlaceIn) -> PlaceOut:
        try:
            return db.add_place(place.name, place.lon, place.lat, place.is_default)
        except DuplicateName:
            raise HTTPException(409, f"Место «{place.name}» уже есть.") from None

    @app.put("/api/places/{place_id}")
    def put_place(place_id: int, change: PlaceChange) -> PlaceOut:
        try:
            place = db.update_place(place_id, **change.model_dump())
        except DuplicateName:
            raise HTTPException(409, f"Место «{change.name}» уже есть.") from None
        if place is None:
            raise HTTPException(404, "Такого места нет.")
        return place

    @app.delete("/api/places/{place_id}", status_code=204)
    def delete_place(place_id: int) -> None:
        if not db.delete_place(place_id):
            raise HTTPException(404, "Такого места нет.")

    def importer() -> TrackImport:
        if tracks is None:
            raise HTTPException(404, _NO_UPLOADS)
        return tracks

    @app.get("/api/tracks")
    def get_tracks() -> TracksStatus:
        return importer().status()

    @app.put("/api/tracks/files/{name}", status_code=201)
    async def put_track_file(name: str, request: Request) -> UploadedFile:
        """Store the request body as a ride file to be processed; an archive may be large, so it is not held in memory."""
        try:
            with importer().receiving(name) as file:
                async for chunk in request.stream():
                    file.write(chunk)
        except Refused as refusal:
            raise HTTPException(refusal.status, str(refusal)) from None
        return UploadedFile(name=name, size=file.size)

    @app.post("/api/tracks/import", status_code=202)
    def post_tracks_import() -> TracksStatus:
        """Start processing everything uploaded; the rides appear in routes once it is done."""
        if not importer().start():
            raise HTTPException(409, "Обработка уже идёт.")
        return importer().status()

    if static_dir is not None and (static_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static_dir, html=True))
    return app


class _RidesFile:
    """Ride counts on disk, replaced as a whole by `rpa match`; a replaced file is read into the engine."""

    def __init__(self, path: Path, engine: Engine) -> None:
        self.path = path
        self.engine = engine
        self._lock = threading.Lock()
        self._seen = self._stamp()

    def _stamp(self) -> tuple[int, int] | None:
        try:
            status = self.path.stat()
        except OSError:
            return None
        return status.st_mtime_ns, status.st_ino

    def refresh(self) -> bool:
        """True if the engine got new counts. A vanished or unusable file leaves the counts as they were."""
        stamp = self._stamp()
        if stamp is None or stamp == self._seen:
            return False
        with self._lock:
            if stamp == self._seen:
                return False
            self._seen = stamp
            try:
                self.engine.set_rides(np.load(self.path))
            except (OSError, ValueError) as error:  # most likely counted for a graph built since this server started
                logging.getLogger("uvicorn.error").warning("Заезды из %s не подошли: %s", self.path, error)
                return False
            return True


def _gpx(body: CandidateBody, name: str | None, file_name: str) -> Response:
    profile = body.profile and (np.array(body.profile.distance_m), np.array(body.profile.elevation_m))
    return Response(
        to_gpx(np.array(body.geometry.coordinates).reshape(-1, 2), name or body.title, profile),
        media_type="application/gpx+xml",
        headers={"Content-Disposition": f'attachment; filename="{file_name}.gpx"'},
    )


def _meta(engine: Engine) -> Meta:
    g = engine.g
    lonlat = np.asarray(g.node_lonlat)
    length = engine.length.astype(np.float64)
    ridden_m = float(length[engine.rides > 0].sum()) if engine.rides is not None else 0.0
    return Meta(
        nodes=g.n_nodes,
        edges=g.n_edges,
        road_km=float(length.sum()) / 1000,
        bbox=(*lonlat.min(axis=0).tolist(), *lonlat.max(axis=0).tolist()),
        has_rides=engine.rides is not None,
        ridden_km=ridden_m / 1000,
        has_water=engine.at_water is not None,
        has_terrain=engine.forest is not None,
        has_gazetteer=False,
        has_llm=False,
        can_import_tracks=False,
    )


def _candidate_line(number: int, candidate: CandidateBody) -> str:
    """One offered route in words, for the model to relate a follow-up to."""
    measured = candidate.features
    parts = [f"{number}. {candidate.title}"]
    if candidate.ascent_m is not None:
        parts.append(f"набор {candidate.ascent_m:.0f} м")
    parts.append(f"лес {measured.get('forest_share', 0):.0%}")
    parts.append(f"знакомые дороги {measured.get('ridden_share', 0):.0%}")
    parts.append(f"крупные дороги {measured.get('major_share', 0):.0%}")
    return "; ".join(parts)


def _check_point(engine: Engine, point: Point, name: str) -> list[str]:
    """Warnings about where `point` lands on the map; refuses a point that is off it."""
    _, distance = engine.snapper.nearest_node(point.lon, point.lat)
    if distance > MAX_SNAP_M:
        raise HTTPException(
            422, f"Точка «{name}» вне карты: до ближайшей дороги {distance / 1000:.0f} км. Выберите точку в регионе."
        )
    if distance > FAR_SNAP_M:
        return [f"Точка «{name}» далеко от дорог ({distance:.0f} м): маршрут начнётся от ближайшей."]
    return []


def _body(candidate: Candidate, spec: RouteSpec) -> CandidateBody:
    path, measured = candidate.path, candidate.features
    km = path.length_m / 1000
    title = f"{'Кольцо' if spec.shape == 'loop' else 'Маршрут'} {km:.1f} км"
    if candidate.via_label:
        title += f" · {candidate.via_label}"
    return CandidateBody(
        title=title,
        geometry=LineString(coordinates=np.round(path.coords, 6).tolist()),
        length_km=round(km, 3),
        ascent_m=round(measured.ascent_m) if candidate.profile is not None else None,
        score=round(candidate.score, 1),
        note=candidate.note,
        via_label=candidate.via_label,
        features={name: round(value, 4) for name, value in asdict(measured).items()},
        km_by_highway={name: round(value, 3) for name, value in path.km_by_highway.items()},
        km_by_surface={name: round(value, 3) for name, value in path.km_by_surface.items()},
        profile=_thin_profile(candidate.profile),
    )


def _thin_profile(profile: tuple[np.ndarray, np.ndarray] | None) -> Profile | None:
    """At most MAX_PROFILE_POINTS evenly picked samples, always with the first and the last."""
    if profile is None:
        return None
    distance, height = profile
    if len(distance) > MAX_PROFILE_POINTS:
        pick = np.unique(np.round(np.linspace(0, len(distance) - 1, MAX_PROFILE_POINTS)).astype(np.int64))
        distance, height = distance[pick], height[pick]
    return Profile(distance_m=np.round(distance, 1).tolist(), elevation_m=np.round(height, 1).tolist())
