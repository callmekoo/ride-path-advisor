import gzip
import struct
import zipfile

import numpy as np
import pytest

from rpa.tracks.clean import clean
from rpa.tracks.ingest import import_inbox
from rpa.tracks.parse import kind_of, parse_track
from rpa.tracks.store import load_tracks, save_tracks

BBOX = (59.0, 56.0, 62.0, 58.0)
M_PER_DEG_LAT = 111_195.0


def line(n: int, step_m: float = 20.0, lon: float = 60.0, lat: float = 56.8) -> np.ndarray:
    """n points heading north, step_m apart."""
    return np.stack([np.full(n, lon), lat + np.arange(n) * step_m / M_PER_DEG_LAT], axis=1)


def gpx(points: np.ndarray) -> bytes:
    body = "".join(
        f'<trkpt lat="{lat:.7f}" lon="{lon:.7f}"><time>2024-05-01T10:00:{i % 60:02d}Z</time></trkpt>'
        for i, (lon, lat) in enumerate(points)
    )
    return (
        '<?xml version="1.0"?><gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        f"{body}</trkseg></trk></gpx>"
    ).encode()


TCX = b"""   <?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">
<Activities><Activity Sport="Biking"><Lap><Track>
<Trackpoint><Time>2024-05-01T10:00:00Z</Time>
  <Position><LatitudeDegrees>56.8</LatitudeDegrees><LongitudeDegrees>60.0</LongitudeDegrees></Position></Trackpoint>
<Trackpoint><Time>2024-05-01T10:00:05Z</Time><HeartRateBpm><Value>120</Value></HeartRateBpm></Trackpoint>
<Trackpoint><Time>2024-05-01T10:00:10Z</Time>
  <Position><LatitudeDegrees>56.801</LatitudeDegrees><LongitudeDegrees>60.001</LongitudeDegrees></Position></Trackpoint>
</Track></Lap></Activity></Activities></TrainingCenterDatabase>"""


def fit(points: list[tuple[float, float]]) -> bytes:
    """A minimal FIT file: one definition of the `record` message, then one record per point."""

    def crc16(data: bytes) -> int:
        table = [0x0000, 0xCC01, 0xD801, 0x1400, 0xF001, 0x3C00, 0x2800, 0xE401,
                 0xA001, 0x6C00, 0x7800, 0xB401, 0x5000, 0x9C01, 0x8801, 0x4400]  # fmt: skip
        crc = 0
        for byte in data:
            for nibble in (byte & 0xF, byte >> 4):
                crc = (crc >> 4) ^ table[crc & 0xF] ^ table[nibble]
        return crc

    # timestamp (uint32), position_lat (sint32), position_long (sint32) of global message 20.
    body = struct.pack("<BBBHB", 0x40, 0, 0, 20, 3) + bytes([253, 4, 0x86, 0, 4, 0x85, 1, 4, 0x85])
    for i, (lon, lat) in enumerate(points):
        to_semicircles = 2**31 / 180
        body += struct.pack("<BIii", 0, 1_000_000_000 + i, round(lat * to_semicircles), round(lon * to_semicircles))
    header = struct.pack("<BBHI4sH", 14, 0x20, 2100, len(body), b".FIT", 0)
    return header + body + struct.pack("<H", crc16(header + body))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("activities/1.gpx", ("gpx", False)),
        ("activities/2.GPX.gz", ("gpx", True)),
        ("3.fit.gz", ("fit", True)),
        ("4.tcx", ("tcx", False)),
        ("photo.jpg", None),
        ("", None),
    ],
)
def test_kind_of(name, expected):
    assert kind_of(name) == expected


def test_parse_gpx():
    track = parse_track(gpx(line(3)), "gpx")
    np.testing.assert_allclose(track.lonlat, line(3), atol=1e-7)
    assert track.time[1] - track.time[0] == 1.0


def test_parse_gzipped_gpx():
    assert len(parse_track(gzip.compress(gpx(line(3))), "gpx", gzipped=True).lonlat) == 3


def test_parse_tcx_skips_points_without_position_and_tolerates_leading_whitespace():
    track = parse_track(TCX, "tcx")
    np.testing.assert_allclose(track.lonlat, [[60.0, 56.8], [60.001, 56.801]])
    assert track.time[1] - track.time[0] == 10.0


def test_parse_fit_converts_semicircles():
    track = parse_track(fit([(60.0, 56.8), (60.001, 56.801)]), "fit")
    np.testing.assert_allclose(track.lonlat, [[60.0, 56.8], [60.001, 56.801]], atol=1e-6)
    assert track.time[1] - track.time[0] == 1.0


def test_parse_file_without_points():
    track = parse_track(gpx(np.empty((0, 2))), "gpx")
    assert track.lonlat.shape == (0, 2)


def test_clean_keeps_a_plain_ride_in_one_segment():
    segments = clean(line(100), BBOX)
    assert len(segments) == 1
    np.testing.assert_allclose(segments[0][0], line(100)[0])
    np.testing.assert_allclose(segments[0][-1], line(100)[-1])


def test_clean_thins_dense_points_to_about_ten_metres():
    segments = clean(line(1000, step_m=1.0), BBOX)
    assert 95 <= len(segments[0]) <= 105


def test_clean_removes_a_lone_spike_without_cutting_the_ride():
    points = line(100)
    points[50] = [60.5, 57.5]
    segments = clean(points, BBOX)
    assert len(segments) == 1
    assert np.abs(segments[0][:, 0] - 60.0).max() < 1e-9


def test_clean_cuts_at_a_gap():
    points = np.concatenate([line(50), line(50, lat=56.9)])
    assert [len(s) for s in clean(points, BBOX)] == [50, 50]


def test_clean_cuts_where_the_ride_leaves_the_region():
    points = line(150)
    points[50:100, 0] = 70.0  # a kilometre of the ride lies outside the box
    assert [len(s) for s in clean(points, BBOX)] == [50, 50]


def test_clean_drops_null_island_and_non_finite_points():
    points = line(100)
    points[10] = [0.0, 0.0]
    points[20] = [np.nan, np.nan]
    assert sum(len(s) for s in clean(points, BBOX)) == 98


def test_clean_drops_scraps():
    assert clean(line(4), BBOX) == []  # too few points
    assert clean(line(10, step_m=10.0), BBOX) == []  # under 300 m
    assert clean(np.empty((0, 2)), BBOX) == []


CSV_HEAD = ",".join(f"колонка {i}" for i in range(14))


def csv_row(activity_id: str, activity_type: str, filename: str, commute: str = "false") -> str:
    cells = [""] * 14
    cells[0], cells[3], cells[9], cells[11], cells[12] = activity_id, activity_type, commute, "Байк", filename
    return ",".join(cells)


def strava_zip(path, rows: list[str], files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("activities.csv", "﻿" + "\n".join([CSV_HEAD, *rows]) + "\n")
        for name, data in files.items():
            archive.writestr(name, data)


@pytest.fixture
def inbox(tmp_path):
    folder = tmp_path / "inbox"
    folder.mkdir()
    strava_zip(
        folder / "export_1.zip",
        [
            csv_row("11", "Велосипед", "activities/11.gpx", commute="true"),
            csv_row("12", "Ride", "activities/12.fit.gz"),
            csv_row("13", "Ходьба", "activities/13.gpx"),
            csv_row("14", "Виртуальный заезд", "activities/14.gpx"),
            csv_row("15", "Велосипед", ""),
            csv_row("16", "Велосипед", "activities/16.gpx"),
            csv_row("17", "Велосипед", "activities/17.gpx"),
        ],
        {
            "activities/11.gpx": gpx(line(100)),
            "activities/12.fit.gz": gzip.compress(fit([tuple(p) for p in line(60, lon=60.1)])),
            "activities/13.gpx": gpx(line(100)),
            "activities/14.gpx": gpx(line(100)),
            "activities/16.gpx": gpx(line(100, lon=30.0)),  # a trip far away
            "activities/17.gpx": b"<gpx><broken",
        },
    )
    # The same activity again in a later export, and a loose file.
    strava_zip(
        folder / "export_2.zip",
        [csv_row("11", "Велосипед", "activities/11.gpx")],
        {"activities/11.gpx": gpx(line(100))},
    )
    (folder / "sunday.gpx").write_bytes(gpx(line(80, lon=60.2)))
    (folder / "notes.txt").write_text("not a track")
    (folder / ".monday.gpx").write_bytes(gpx(line(80, lon=60.3)))  # hidden: still being uploaded, say
    return folder


def test_import_inbox(inbox):
    tracks, report = import_inbox(inbox, BBOX, log=lambda _: None)
    assert [t["id"] for t in tracks.tracks] == ["strava:11", "strava:12", "file:sunday.gpx"]
    assert tracks.tracks[0] == {
        "id": "strava:11",
        "source": "export_1.zip",
        "type": "Велосипед",
        "gear": "Байк",
        "commute": True,
        "date": "2024-05-01",
    }
    assert tracks.n_segments == 3
    np.testing.assert_allclose(tracks.segment(1)[0], [60.1, 56.8], atol=1e-6)

    assert report.files == 5
    assert report.imported == 3
    assert report.outside_region == 1
    assert dict(report.skipped_types) == {"Ходьба": 1, "Виртуальный заезд": 1}
    assert len(report.unreadable) == 1 and "strava:17" in report.unreadable[0]
    assert report.km == pytest.approx((99 + 59 + 79) * 0.02, rel=0.01)


def test_tracks_round_trip(inbox, tmp_path):
    tracks, _ = import_inbox(inbox, BBOX, log=lambda _: None)
    save_tracks(tracks, tmp_path / "tracks")
    save_tracks(tracks, tmp_path / "tracks")  # overwriting is fine
    loaded = load_tracks(tmp_path / "tracks")
    np.testing.assert_array_equal(loaded.points, tracks.points)
    np.testing.assert_array_equal(loaded.seg_off, tracks.seg_off)
    np.testing.assert_array_equal(loaded.seg_track, tracks.seg_track)
    assert loaded.tracks == tracks.tracks
