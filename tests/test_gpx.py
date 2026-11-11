import xml.etree.ElementTree as ET

import numpy as np

from rpa.routing.gpx import to_gpx

NS = {"g": "http://www.topografix.com/GPX/1/1"}

# Three points on a meridian, evenly spaced.
LINE = np.array([[60.0, 56.80], [60.0, 56.81], [60.0, 56.82]])


def points(text: str) -> list[ET.Element]:
    return ET.fromstring(text).findall("g:trk/g:trkseg/g:trkpt", NS)


def test_gpx_is_a_version_1_1_track_with_every_point():
    root = ET.fromstring(to_gpx(LINE, "Кольцо 45 км", None))

    assert root.tag == "{http://www.topografix.com/GPX/1/1}gpx"
    assert root.get("version") == "1.1"
    assert root.get("creator")
    found = root.findall("g:trk/g:trkseg/g:trkpt", NS)
    assert [(float(p.get("lon")), float(p.get("lat"))) for p in found] == [tuple(row) for row in LINE.tolist()]


def test_name_survives_characters_that_need_escaping():
    name = 'К воде <Шарташ> & "обратно"'
    root = ET.fromstring(to_gpx(LINE, name, None))
    assert root.find("g:trk/g:name", NS).text == name


def test_no_elevation_without_a_profile():
    assert all(p.find("g:ele", NS) is None for p in points(to_gpx(LINE, "x", None)))


def test_elevation_is_interpolated_along_the_line():
    # The profile is sampled on its own distance scale, a little longer than the line itself.
    profile = (np.array([0.0, 500.0, 2300.0]), np.array([100.0, 100.0 + 100 * 500 / 2300, 200.0]))
    elevations = [float(p.find("g:ele", NS).text) for p in points(to_gpx(LINE, "x", profile))]
    assert elevations == [100.0, 150.0, 200.0]


def test_empty_route_is_still_valid_gpx():
    assert points(to_gpx(np.empty((0, 2)), "x", None)) == []
