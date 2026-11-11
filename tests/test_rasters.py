import numpy as np
import pytest
from helpers import edge_index
from rasterio.transform import from_origin

from rpa.enrich.rasters import climb_m, edge_ascent, edge_samples, elevation, forest_share, sample_tiles

PIXEL = 1 / 1200  # degrees, about 90 m north-south


def tile_from(function, size_deg: int = 1):
    """A tile reader whose rasters hold function(lon, lat) at pixel centres."""

    def read(lat0: int, lon0: int):
        n = round(size_deg / PIXEL)
        lon = lon0 + (np.arange(n) + 0.5) * PIXEL
        lat = lat0 + size_deg - (np.arange(n) + 0.5) * PIXEL
        grid_lon, grid_lat = np.meshgrid(lon, lat)
        return function(grid_lon, grid_lat).astype(np.float32), from_origin(lon0, lat0 + size_deg, PIXEL, PIXEL)

    return read


def quiet(_: str) -> None:
    pass


def test_samples_are_evenly_spaced_and_include_both_ends(micro):
    lonlat, off = edge_samples(micro)
    assert len(off) == micro.n_edges + 1
    e = edge_index(micro, 3, 6)  # 222 m straight north through node 5
    points = lonlat[off[e] : off[e + 1]]
    assert len(points) == 6  # every 44 m
    np.testing.assert_allclose(points[0], [60.002, 56.800], atol=1e-9)
    np.testing.assert_allclose(points[-1], [60.002, 56.802], atol=1e-9)
    np.testing.assert_allclose(np.diff(points[:, 1]), 0.0004, atol=1e-9)
    assert (np.diff(off.astype(np.int64)) >= 2).all()


def test_samples_follow_bends(micro):
    lonlat, off = edge_samples(micro)
    e = edge_index(micro, 12, 6)  # west 61 m to node 13, then south 111 m to node 6
    points = lonlat[off[e] : off[e + 1]]
    assert points[:, 0].min() == pytest.approx(60.002, abs=1e-9)
    assert (np.abs(points[:, 0] - 60.002) < 1e-9).sum() >= 3  # the southward leg


def test_bilinear_sampling_of_a_plane_is_exact():
    lonlat = np.array([[60.3217, 56.4711], [60.9, 56.02]])
    plane = tile_from(lambda lon, lat: 100 * lon + 10 * lat)
    values = sample_tiles(lonlat, 1, plane, bilinear=True, log=quiet)
    np.testing.assert_allclose(values, 100 * lonlat[:, 0] + 10 * lonlat[:, 1], rtol=1e-6)


def test_sampling_spans_tiles_and_tolerates_missing_ones():
    lonlat = np.array([[60.5, 56.5], [61.5, 56.5], [62.5, 56.5]])

    def reader(lat0: int, lon0: int):
        return None if lon0 == 61 else tile_from(lambda lon, lat: np.full_like(lon, lon0))(lat0, lon0)

    assert sample_tiles(lonlat, 1, reader, bilinear=False, log=quiet).tolist() == [60.0, 0.0, 62.0]


def test_ascent_on_a_steady_slope(micro):
    # The ground rises 5 m for every 100 m northwards.
    slope = tile_from(lambda lon, lat: (lat - 56.8) * 111_195 * 0.05 + 200)
    lonlat, off = edge_samples(micro)
    ele = elevation(lonlat, slope, log=quiet)
    up, down = edge_ascent(off, ele)

    north = edge_index(micro, 3, 6)  # 222 m north
    assert ele[off[north]] == pytest.approx(2000, abs=1)  # decimetres
    assert up[north] == pytest.approx(11.1, abs=0.3)
    assert down[north] == pytest.approx(0.0, abs=0.05)
    east = edge_index(micro, 1, 3)
    assert up[east] == pytest.approx(0.0, abs=0.05) and down[east] == pytest.approx(0.0, abs=0.05)
    south = edge_index(micro, 12, 6)  # ends 111 m further south than it starts
    assert up[south] == pytest.approx(0.0, abs=0.05)
    assert down[south] == pytest.approx(5.55, abs=0.3)


def test_forest_share(micro):
    # Forest covers everything north of 56.801.
    north_forest = tile_from(lambda lon, lat: lat > 56.801, size_deg=3)
    lonlat, off = edge_samples(micro)
    forest = forest_share(lonlat, off, north_forest, log=quiet) / 255
    assert forest[edge_index(micro, 1, 3)] == 0.0
    assert forest[edge_index(micro, 12, 14)] == 1.0
    assert forest[edge_index(micro, 3, 6)] == pytest.approx(0.5, abs=0.2)


def test_climb_ignores_ripples_and_counts_real_hills():
    flat_but_noisy = 200 + np.tile([0.0, 3.0, -2.0, 1.0], 50)
    assert climb_m(flat_but_noisy) == 0.0
    hill = np.concatenate([np.linspace(200, 260, 60), np.linspace(260, 230, 30), np.linspace(230, 250, 20)])
    assert climb_m(hill) == pytest.approx(80, abs=6)
    assert climb_m(np.array([200.0])) == 0.0
