import numpy as np
import pytest

from rpa.match.sizes import ride_lengths_km, ride_sizes_km
from rpa.tracks.store import make_trackset

M_PER_DEG_LAT = 111_195.0


def north(km: float, lon: float = 60.0, lat: float = 56.8) -> np.ndarray:
    """A straight segment heading north, `km` long, with a point every 500 m."""
    n = max(int(km * 2), 1) + 1
    return np.stack([np.full(n, lon), lat + np.linspace(0, km * 1000, n) / M_PER_DEG_LAT], axis=1)


def trackset(rides: list[tuple[float, bool]]):
    tracks = [{"id": str(i), "commute": commute} for i, (_, commute) in enumerate(rides)]
    return make_trackset(tracks, [[north(km)] for km, _ in rides])


def lengths(*km: float):
    return trackset([(length, False) for length in km])


def test_ride_length_is_the_sum_of_segments_without_the_gap_between_them():
    tracks = make_trackset([{"id": "a"}], [[north(10, lat=56.0), north(5, lat=57.0)]])
    assert ride_lengths_km(tracks) == pytest.approx([15.0], rel=1e-3)


def test_sizes_are_rounded_quartiles_of_ride_length():
    # 21 rides, 20..60 km evenly: quartiles 30, 40, 50.
    sizes = ride_sizes_km(lengths(*np.linspace(20, 60, 21)))
    assert sizes == (30.0, 40.0, 50.0)


def test_sizes_are_rounded_to_five_km():
    sizes = ride_sizes_km(lengths(*np.linspace(21, 71, 21)))  # quartiles 33.5, 46, 58.5
    assert sizes == (35.0, 45.0, 60.0)


def test_commutes_and_short_rides_are_left_out():
    rides = [(length, False) for length in np.linspace(20, 60, 21)]
    rides += [(100.0, True)] * 30 + [(5.0, False)] * 30 + [(9.9, False)] * 5
    assert ride_sizes_km(trackset(rides)) == (30.0, 40.0, 50.0)


def test_too_few_rides_give_nothing():
    assert ride_sizes_km(lengths(*np.linspace(20, 60, 19))) is None
    # Enough rides, but only 19 of them count.
    assert ride_sizes_km(trackset([(length, False) for length in np.linspace(20, 60, 19)] + [(80.0, True)])) is None


def test_sizes_that_stick_together_give_nothing():
    assert ride_sizes_km(lengths(*np.linspace(40, 50, 40))) is None  # 42.5 / 45 / 47.5


def test_a_step_of_exactly_twenty_percent_is_enough():
    # Quartiles 25 / 30 / 36 -> 25 / 30 / 35: 30 = 1.2 * 25 passes, 35 = 1.17 * 30 does not.
    assert ride_sizes_km(lengths(*([25.0] * 6 + [30.0] * 8 + [36.0] * 6 + [60.0]))) is None
    # 25 / 30 / 40: both steps reach 20 %.
    assert ride_sizes_km(lengths(*([25.0] * 6 + [30.0] * 8 + [40.0] * 7))) == (25.0, 30.0, 40.0)
