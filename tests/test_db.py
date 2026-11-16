import json
import sqlite3

import pytest

from rpa.api.db import KEEP_REQUESTS, SCHEMA_VERSION, Db, DuplicateName


@pytest.fixture
def db(tmp_path) -> Db:
    return Db(tmp_path / "rpa.sqlite")


def test_first_place_becomes_the_default(db):
    home = db.add_place("дом", 60.6, 56.84)
    work = db.add_place("работа", 60.5, 56.8)

    assert home["is_default"] and not work["is_default"]
    assert db.default_place()["name"] == "дом"
    assert [place["name"] for place in db.places()] == ["дом", "работа"]


def test_there_is_exactly_one_default(db):
    db.add_place("дом", 60.6, 56.84)
    dacha = db.add_place("дача", 60.1, 56.9, is_default=True)
    assert [place["is_default"] for place in db.places()] == [False, True]

    first = db.places()[0]
    db.update_place(first["id"], is_default=True)
    assert [place["is_default"] for place in db.places()] == [True, False]
    assert db.default_place()["id"] != dacha["id"]


def test_the_default_cannot_simply_be_switched_off(db):
    home = db.add_place("дом", 60.6, 56.84)
    assert db.update_place(home["id"], is_default=False)["is_default"]


def test_deleting_the_default_passes_it_to_the_oldest_remaining(db):
    home = db.add_place("дом", 60.6, 56.84)
    db.add_place("работа", 60.5, 56.8)
    db.add_place("дача", 60.1, 56.9)

    assert db.delete_place(home["id"])
    assert db.default_place()["name"] == "работа"
    assert not db.delete_place(home["id"])


def test_no_default_without_places(db):
    assert db.default_place() is None
    assert db.places() == []


def test_names_are_unique_whatever_the_case(db):
    db.add_place("Дом", 60.6, 56.84)
    other = db.add_place("работа", 60.5, 56.8)
    with pytest.raises(DuplicateName):
        db.add_place("дом", 60.0, 56.0)
    with pytest.raises(DuplicateName):
        db.update_place(other["id"], name="ДОМ")


def test_update_changes_only_what_is_given(db):
    home = db.add_place("дом", 60.6, 56.84)
    moved = db.update_place(home["id"], lon=60.7)

    assert (moved["name"], moved["lon"], moved["lat"]) == ("дом", 60.7, 56.84)
    assert db.update_place(home["id"], name="Дом")["name"] == "Дом"  # renaming to itself is not a duplicate
    assert db.update_place(999, name="x") is None


def test_candidates_are_stored_with_their_request(db):
    request, ids = db.save_request('{"shape": "loop"}', ["first", "second"])
    other, other_ids = db.save_request("{}", ["third"])

    assert len(set(ids + other_ids)) == 3
    assert db.candidate(ids[1]) == "second"
    assert db.candidate(10_000) is None
    assert db.request(request) == {"spec": '{"shape": "loop"}', "candidates": ["first", "second"]}
    assert db.request(10_000) is None


def test_old_requests_are_dropped_with_their_candidates(db):
    first, first_ids = db.save_request("{}", ["a"])
    for _ in range(KEEP_REQUESTS):
        last, last_ids = db.save_request("{}", ["b"])

    assert db.request(first) is None and db.candidate(first_ids[0]) is None
    assert db.candidate(last_ids[0]) == "b"


def test_data_survives_reopening(tmp_path):
    path = tmp_path / "nested" / "rpa.sqlite"
    Db(path).add_place("дом", 60.6, 56.84)
    _, ids = Db(path).save_request("{}", ["a"])

    assert Db(path).default_place()["name"] == "дом"
    assert Db(path).candidate(ids[0]) == "a"


def body(title: str, km: float, ascent: float | None = None) -> str:
    return json.dumps({"title": title, "length_km": km, "ascent_m": ascent})


def test_a_saved_route_keeps_the_candidate_and_its_request(db):
    _, ids = db.save_request('{"shape": "loop"}', [body("Кольцо 45 км", 45.2, 310), body("Кольцо 44 км", 44.0)])

    saved = db.save_route(ids[0], "вокруг озера")
    untitled = db.save_route(ids[1])

    assert saved == {"id": saved["id"], "name": "вокруг озера", "created": saved["created"], "length_km": 45.2, "ascent_m": 310}
    assert (untitled["name"], untitled["ascent_m"]) == ("Кольцо 44 км", None)
    assert db.route(saved["id"]) == saved | {"spec": '{"shape": "loop"}', "body": body("Кольцо 45 км", 45.2, 310)}
    assert db.save_route(10_000) is None and db.route(10_000) is None


def test_saved_routes_outlive_their_requests(db):
    _, ids = db.save_request("{}", [body("Кольцо", 45.0)])
    saved = db.save_route(ids[0])
    for _ in range(KEEP_REQUESTS):
        db.save_request("{}", [])

    assert db.candidate(ids[0]) is None
    assert db.route(saved["id"])["body"] == body("Кольцо", 45.0)


def test_saved_routes_are_listed_newest_first_and_can_be_deleted(db):
    _, ids = db.save_request("{}", [body("первый", 10.0), body("второй", 20.0)])
    first, second = db.save_route(ids[0]), db.save_route(ids[1])

    assert db.routes() == [second, first]
    assert db.delete_route(first["id"]) and not db.delete_route(first["id"])
    assert db.routes() == [second]


def _set_version(path, version: int) -> None:
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()


def _version(path) -> int:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()


@pytest.mark.parametrize("old_version", [0, 1])
def test_an_older_schema_loses_requests_and_routes_but_keeps_places(tmp_path, old_version):
    path = tmp_path / "rpa.sqlite"
    old = Db(path)
    old.add_place("дом", 60.6, 56.84)
    _, ids = old.save_request("{}", [body("Кольцо", 45.0)])
    old.save_route(ids[0])
    _set_version(path, old_version)

    reopened = Db(path)

    assert [place["name"] for place in reopened.places()] == ["дом"]
    assert reopened.candidate(ids[0]) is None and reopened.request(1) is None and reopened.routes() == []
    assert _version(path) == SCHEMA_VERSION == 2


def test_reopening_a_current_database_erases_nothing(tmp_path):
    path = tmp_path / "rpa.sqlite"
    first = Db(path)
    first.add_place("дом", 60.6, 56.84)
    request_id, ids = first.save_request("{}", [body("Кольцо", 45.0)])
    saved = first.save_route(ids[0])

    again = Db(path)

    assert again.request(request_id) is not None and again.routes() == [saved] and len(again.places()) == 1


def test_a_new_database_starts_at_the_current_version(tmp_path):
    Db(tmp_path / "rpa.sqlite")
    assert _version(tmp_path / "rpa.sqlite") == SCHEMA_VERSION
