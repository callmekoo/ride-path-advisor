"""The HTTP API: routes for a RouteSpec, GPX export, saved places.

Handlers are plain functions, so FastAPI runs them in its thread pool; the search
kernel releases the GIL, which lets requests overlap.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response

from rpa.api.db import Db, DuplicateName
from rpa.api.schemas import (
    CandidateBody,
    CandidateOut,
    LineString,
    Meta,
    PlaceChange,
    PlaceIn,
    PlaceOut,
    Profile,
    RouteResponse,
)
from rpa.routing.engine import Engine
from rpa.routing.generate import generate
from rpa.routing.gpx import to_gpx
from rpa.routing.score import Candidate
from rpa.routing.spec import Point, RouteSpec

# Beyond FAR_SNAP_M a snapped point probably is not where the user meant; beyond
# MAX_SNAP_M it is off the map and a route from the nearest road would mislead.
FAR_SNAP_M = 500.0
MAX_SNAP_M = 2000.0
MAX_PROFILE_POINTS = 500

_NO_WATER_DATA = "В графе нет данных о воде: выполните rpa enrich."
_NOTHING_FOUND = "Не удалось построить маршрут с такими условиями. Попробуйте другую длину или более мягкие ограничения."


def create_app(engine: Engine, db_path: Path) -> FastAPI:
    app = FastAPI(title="ride-path-advisor")
    db = Db(db_path)
    meta = _meta(engine)

    @app.get("/api/meta")
    def get_meta() -> Meta:
        return meta

    @app.post("/api/route")
    def post_route(spec: RouteSpec) -> RouteResponse:
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

    @app.get("/api/candidates/{candidate_id}.gpx")
    def get_gpx(candidate_id: int) -> Response:
        stored = db.candidate(candidate_id)
        if stored is None:
            raise HTTPException(404, "Маршрут не найден: возможно, он устарел. Постройте его заново.")
        body = CandidateBody.model_validate_json(stored)
        profile = body.profile and (np.array(body.profile.distance_m), np.array(body.profile.elevation_m))
        return Response(
            to_gpx(np.array(body.geometry.coordinates).reshape(-1, 2), body.title, profile),
            media_type="application/gpx+xml",
            headers={"Content-Disposition": f'attachment; filename="route-{candidate_id}.gpx"'},
        )

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

    return app


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
    )


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
