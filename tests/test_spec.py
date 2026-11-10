import pytest

from rpa.routing.cost import Prefs
from rpa.routing.spec import LengthKm, Point, RouteSpec

START = Point(lon=60.6, lat=56.84)
END = Point(lon=59.93, lat=56.80, label="Ревда")


def test_around_gives_ten_percent_either_way():
    length = LengthKm.around(45)
    assert (length.min, length.target, length.max) == pytest.approx((40.5, 45, 49.5))


def test_between_targets_the_middle():
    length = LengthKm.between(100, 150)
    assert (length.min, length.target, length.max) == (100, 125, 150)


@pytest.mark.parametrize("bounds", [(50, 45, 60), (40, 45, 44), (0, 45, 50), (-1, 1, 2)])
def test_length_must_be_ordered_and_positive(bounds):
    low, target, high = bounds
    with pytest.raises(ValueError):
        LengthKm(min=low, target=target, max=high)


def test_loop_spec():
    spec = RouteSpec(start=START, shape="loop", length_km=LengthKm.around(45))
    assert spec.n_candidates == 3
    assert spec.prefs == Prefs()


def test_loop_needs_a_length():
    with pytest.raises(ValueError, match="length"):
        RouteSpec(start=START, shape="loop")


def test_loop_takes_no_end_point():
    with pytest.raises(ValueError, match="end"):
        RouteSpec(start=START, end=END, shape="loop", length_km=LengthKm.around(45))


def test_p2p_needs_an_end_point():
    with pytest.raises(ValueError, match="end"):
        RouteSpec(start=START, shape="p2p")


def test_p2p_length_is_optional():
    assert RouteSpec(start=START, end=END, shape="p2p").length_km is None


def test_spec_round_trips_through_json():
    spec = RouteSpec(
        start=START, end=END, shape="p2p", length_km=LengthKm.around(80), prefs=Prefs(avoid_major=0.9, surface="paved")
    )
    assert RouteSpec.model_validate_json(spec.model_dump_json()) == spec


def test_candidate_count_is_bounded():
    with pytest.raises(ValueError):
        RouteSpec(start=START, shape="loop", length_km=LengthKm.around(45), n_candidates=0)
