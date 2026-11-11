import sys

import pytest

from rpa.api.jobs import MAX_UPLOAD_BYTES, Refused, TrackImport


def importer(tmp_path, code: str = "print('импортировано заездов: 3')") -> TrackImport:
    return TrackImport(tmp_path / "inbox", tmp_path / "jobs" / "tracks.log", [sys.executable, "-c", code])


def upload(tracks: TrackImport, name: str, *chunks: bytes) -> int:
    with tracks.receiving(name) as file:
        for chunk in chunks:
            file.write(chunk)
    return file.size


def test_an_uploaded_file_lands_in_the_inbox(tmp_path):
    tracks = importer(tmp_path)

    assert upload(tracks, "Утро.GPX", b"<gpx>", b"</gpx>") == 11
    assert (tmp_path / "inbox" / "Утро.GPX").read_bytes() == b"<gpx></gpx>"
    assert tracks.status()["inbox"] == [{"name": "Утро.GPX", "size": 11}]
    assert upload(tracks, "Утро.GPX", b"new") == 3  # the same name replaces the file


@pytest.mark.parametrize("name", ["export.zip", "a.gpx", "a.fit", "a.tcx", "a.gpx.gz", "a.FIT.GZ", "ride 12.tcx.gz"])
def test_track_files_and_archives_are_accepted(tmp_path, name):
    assert upload(importer(tmp_path), name, b"x") == 1


@pytest.mark.parametrize("name", ["notes.txt", "a.gz", ".hidden.gpx", "../up.gpx", "dir/a.gpx", "a.gpx/", "", "x" * 300 + ".gpx"])
def test_other_names_are_refused(tmp_path, name):
    tracks = importer(tmp_path)
    with pytest.raises(Refused) as refusal, tracks.receiving(name) as file:
        file.write(b"x")
    assert refusal.value.status == 400
    assert tracks.status()["inbox"] == []


def test_a_broken_upload_leaves_nothing_behind(tmp_path):
    tracks = importer(tmp_path)
    with pytest.raises(ConnectionError), tracks.receiving("a.gpx") as file:
        file.write(b"half")
        raise ConnectionError
    assert list((tmp_path / "inbox").iterdir()) == []


def test_an_oversized_upload_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr("rpa.api.jobs.MAX_UPLOAD_BYTES", 10)
    tracks = importer(tmp_path)
    with pytest.raises(Refused) as refusal, tracks.receiving("a.gpx") as file:
        file.write(b"123456")
        file.write(b"789012")
    assert refusal.value.status == 413
    assert list((tmp_path / "inbox").iterdir()) == []
    assert MAX_UPLOAD_BYTES > 1 << 30  # a Strava archive of many years fits


def test_processing_runs_the_command_and_keeps_its_output(tmp_path):
    tracks = importer(tmp_path)
    assert tracks.status() == {"state": "idle", "started": None, "finished": None, "log": [], "inbox": []}

    assert tracks.start()
    tracks.wait()
    status = tracks.status()

    assert status["state"] == "done"
    assert status["log"] == ["импортировано заездов: 3"]
    assert status["started"] <= status["finished"] and len(status["started"]) == 19


def test_a_failed_run_says_so_and_shows_why(tmp_path):
    tracks = importer(tmp_path, "import sys; print('строка'); sys.exit('Нет треков')")
    tracks.start()
    tracks.wait()
    status = tracks.status()
    assert status["state"] == "failed"
    assert status["log"] == ["строка", "Нет треков"]


def test_only_the_end_of_a_long_log_is_shown(tmp_path):
    tracks = importer(tmp_path, "for i in range(100): print(i)")
    tracks.start()
    tracks.wait()
    assert tracks.status()["log"] == [str(i) for i in range(80, 100)]


def test_one_run_at_a_time_and_no_uploads_meanwhile(tmp_path):
    tracks = importer(tmp_path, "import time; time.sleep(30)")
    assert tracks.start()
    try:
        assert tracks.status()["state"] == "running"
        assert not tracks.start()
        with pytest.raises(Refused) as refusal, tracks.receiving("a.gpx"):
            pass
        assert refusal.value.status == 409
    finally:
        tracks.stop()
    assert tracks.status()["state"] == "failed"
    assert tracks.start()  # and it can be started again
    tracks.stop()
