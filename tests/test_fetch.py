import json

import pytest
from shapely.geometry import Point, box

from rpa.osm.fetch import buffer_km, osmium_command, parse_poly, write_geojson

POLY = """\
region
1
   60.0   56.0
   62.0   56.0
   62.0   58.0
   60.0   58.0
   60.0   56.0
END
!1h
   60.5   56.5
   61.0   56.5
   61.0   57.0
   60.5   57.0
   60.5   56.5
END
2
   70.0   56.0
   71.0   56.0
   71.0   57.0
   70.0   56.0
END
END
"""


def test_parse_poly_reads_rings_and_holes():
    shape = parse_poly(POLY)
    assert shape.contains(Point(60.2, 56.2))
    assert not shape.contains(Point(60.75, 56.75))  # inside the hole
    assert shape.contains(Point(70.8, 56.3))  # second ring
    assert not shape.contains(Point(65.0, 57.0))


def test_parse_poly_rejects_garbage():
    with pytest.raises(ValueError):
        parse_poly("region\n")


def test_buffer_grows_by_the_requested_distance_in_both_axes():
    square = box(60.0, 56.5, 61.0, 57.5)
    grown = buffer_km(square, 100.0)
    west, south, east, north = grown.bounds
    # 100 km is 0.90 degrees of latitude, and 1.65 degrees of longitude at 57 N.
    assert 57.5 + 0.86 < north < 57.5 + 0.94
    assert 56.5 - 0.94 < south < 56.5 - 0.86
    assert 61.0 + 1.57 < east < 61.0 + 1.73
    assert 60.0 - 1.73 < west < 60.0 - 1.57
    assert grown.contains(square)


def test_write_geojson_produces_a_polygon_feature(tmp_path):
    path = tmp_path / "area.geojson"
    write_geojson(box(60.0, 56.0, 61.0, 57.0), path)
    data = json.loads(path.read_text())
    assert data["type"] == "Feature"
    assert data["geometry"]["type"] == "Polygon"


def test_osmium_command_uses_the_native_binary_when_present(tmp_path):
    command = osmium_command(["merge", "a.pbf", "-o", "m.pbf"], tmp_path, native="/usr/bin/osmium")
    assert command == ["/usr/bin/osmium", "merge", "a.pbf", "-o", "m.pbf"]


def test_osmium_command_falls_back_to_a_container(tmp_path):
    command = osmium_command(["merge", "a.pbf", "-o", "m.pbf"], tmp_path, native=None)
    assert command[:3] == ["podman", "run", "--rm"]
    assert f"{tmp_path.resolve()}:/work:Z" in command
    assert command[-4:] == ["merge", "a.pbf", "-o", "m.pbf"]
