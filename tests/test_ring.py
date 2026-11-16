import math

import numpy as np
import pytest
import shapely

from rpa.routing.ring import (
    MIN_AROUND_RADIUS_M,
    encloses,
    from_local,
    object_polygon,
    ring_anchors,
    to_local,
    winding_number,
)
from rpa.routing.spec import Around, Point

ORIGIN = (60.6, 56.84)
M_PER_DEG_LAT = 111_195.0


def circle(radius: float, n: int = 40, centre=(0.0, 0.0), clockwise: bool = False, turns: int = 1) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi * turns, n * turns, endpoint=False)
    if clockwise:
        angles = -angles
    return np.column_stack([centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles)])


# --- projection ---------------------------------------------------------------


def test_local_projection_round_trips_and_has_metres():
    lonlat = np.array([[60.7, 56.9], [60.5, 56.8]])
    xy = to_local(lonlat, ORIGIN)
    assert xy.shape == (2, 2)
    assert xy[0, 1] == pytest.approx(0.06 * M_PER_DEG_LAT, rel=1e-3)
    assert xy[0, 0] == pytest.approx(0.1 * M_PER_DEG_LAT * math.cos(math.radians(56.87)), rel=1e-2)
    assert from_local(xy, ORIGIN) == pytest.approx(lonlat)


def test_local_projection_takes_a_single_pair():
    assert tuple(to_local((60.6, 56.84), ORIGIN)) == pytest.approx((0.0, 0.0))
    assert tuple(from_local((0.0, 0.0), ORIGIN)) == pytest.approx(ORIGIN)


# --- winding number -----------------------------------------------------------


def test_winding_number_counts_turns():
    inside = (0.0, 0.0)
    assert winding_number(circle(100), inside) == 1
    assert winding_number(circle(100, clockwise=True), inside) == -1
    assert winding_number(circle(100, turns=2), inside) == 2
    assert winding_number(circle(100), (500.0, 0.0)) == 0


def test_winding_number_of_a_there_and_back_line_is_zero():
    line = np.array([[0.0, 0.0], [100.0, 10.0], [200.0, 0.0], [100.0, 10.0]])
    assert winding_number(line, (100.0, 5.0)) == 0


def test_figure_eight_turns_oppositely_around_its_two_centres():
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    eight = np.column_stack([100 * np.sin(t), 100 * np.sin(t) * np.cos(t)])
    left, right = (-50.0, 0.0), (50.0, 0.0)
    assert {abs(winding_number(eight, left)), abs(winding_number(eight, right))} == {1}
    assert winding_number(eight, left) == -winding_number(eight, right)


# --- object polygon -----------------------------------------------------------


def test_polygon_from_an_outline_is_in_local_metres():
    point = Point(lon=60.6, lat=56.84)
    outline = [(60.59, 56.83), (60.61, 56.83), (60.61, 56.85), (60.59, 56.85)]
    polygon = object_polygon(Around(point=point, radius_m=1000, outline=outline))
    minx, miny, maxx, maxy = polygon.bounds
    half_width = 0.01 * M_PER_DEG_LAT * math.cos(math.radians(56.84))
    assert (minx, maxx) == pytest.approx((-half_width, half_width), rel=1e-3)
    assert miny == pytest.approx(-0.01 * M_PER_DEG_LAT, rel=1e-3)
    assert maxy == pytest.approx(0.01 * M_PER_DEG_LAT, rel=1e-3)
    assert polygon.contains(shapely.Point(0, 0))


def test_a_self_touching_outline_is_made_valid():
    point = Point(lon=60.6, lat=56.84)
    bowtie = [(60.59, 56.83), (60.61, 56.85), (60.61, 56.83), (60.59, 56.85)]
    polygon = object_polygon(Around(point=point, radius_m=1000, outline=bowtie))
    assert polygon.is_valid and isinstance(polygon, shapely.Polygon) and polygon.area > 0


def test_polygon_without_an_outline_is_a_sixteen_gon_with_a_minimum_radius():
    point = Point(lon=60.6, lat=56.84)
    big = object_polygon(Around(point=point, radius_m=2000))
    assert len(big.exterior.coords) - 1 == 16
    assert big.exterior.distance(shapely.Point(0, 0)) == pytest.approx(2000 * math.cos(math.pi / 16), rel=1e-6)
    small = object_polygon(Around(point=point, radius_m=10))
    assert max(math.hypot(x, y) for x, y in small.exterior.coords) == pytest.approx(MIN_AROUND_RADIUS_M)


# --- anchors ------------------------------------------------------------------


def test_anchors_of_a_long_rectangle_are_offset_and_evenly_spaced():
    rectangle = shapely.box(-1000, -100, 1000, 100)
    n, offset = 24, 150.0
    anchors = ring_anchors(rectangle, offset, n, 0.0)
    assert anchors.shape == (n, 2)
    for x, y in anchors:
        assert rectangle.distance(shapely.Point(x, y)) == pytest.approx(offset, abs=1.0)
    boundary = rectangle.buffer(offset).exterior
    positions = sorted(boundary.project(shapely.Point(x, y)) for x, y in anchors)
    gaps = np.diff([*positions, positions[0] + boundary.length])
    assert gaps == pytest.approx(boundary.length / n, rel=1e-3)


def test_anchor_phase_shifts_the_start_and_direction_is_counter_clockwise():
    square = shapely.box(-100, -100, 100, 100)
    a = ring_anchors(square, 50.0, 8, 0.0)
    b = ring_anchors(square, 50.0, 8, 0.5 / 8)
    assert not np.allclose(a[0], b[0])
    assert winding_number(a, (0.0, 0.0)) == 1


def test_the_buffer_closes_a_narrow_bay():
    # A U-shaped lake: the bay between the arms is 60 m wide, less than twice the offset.
    lake = shapely.box(-500, -300, 500, 0).union(shapely.box(-500, 0, -30, 300)).union(shapely.box(30, 0, 500, 300))
    anchors = ring_anchors(lake, 100.0, 60, 0.0)
    for x, y in anchors:
        assert not (abs(x) < 30 and y > 0 and y < 300)
        assert lake.distance(shapely.Point(x, y)) >= 100.0 - 1.0


# --- encloses -----------------------------------------------------------------


@pytest.fixture
def pond() -> Around:
    # A square of about 2 km on a side with a vertex per 250 m.
    side = np.linspace(0, 1, 9, endpoint=False)
    lon0, lat0, dlon, dlat = 60.6, 56.84, 0.032, 0.018
    outline = (
        [(lon0 + s * dlon, lat0) for s in side]
        + [(lon0 + dlon, lat0 + s * dlat) for s in side]
        + [(lon0 + dlon - s * dlon, lat0 + dlat) for s in side]
        + [(lon0, lat0 + dlat - s * dlat) for s in side]
    )
    return Around(point=Point(lon=lon0 + dlon / 2, lat=lat0 + dlat / 2), radius_m=1000, outline=outline)


def around_route(pond: Around, margin: float, cut_corner: bool = False) -> list[tuple[float, float]]:
    xy = to_local(np.array(pond.outline), (pond.point.lon, pond.point.lat))
    minx, miny = xy.min(axis=0) - margin
    maxx, maxy = xy.max(axis=0) + margin
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    if cut_corner:
        # Come in diagonally across the lower-left corner: the corner vertices fall outside the loop.
        cx, cy = xy[0]
        corners = [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, cy + 700), (cx + 700, cy + 300)]
    return [tuple(p) for p in from_local(np.array(corners), (pond.point.lon, pond.point.lat))]


def test_a_loop_around_the_object_encloses_it(pond):
    assert encloses(around_route(pond, 200), pond)
    assert encloses(around_route(pond, 200)[::-1], pond)


def test_a_loop_beside_the_object_does_not(pond):
    beside = [(pond.point.lon + 0.05 + dx, pond.point.lat + dy) for dx, dy in [(0, 0), (0.02, 0), (0.02, 0.02), (0, 0.02)]]
    assert not encloses(beside, pond)


def test_a_loop_cutting_a_corner_does_not(pond):
    assert not encloses(around_route(pond, 200, cut_corner=True), pond)


def test_there_and_back_does_not(pond):
    there = [(pond.point.lon - 0.05 + i * 0.01, pond.point.lat - 0.03) for i in range(11)]
    assert not encloses(there + there[::-1], pond)
