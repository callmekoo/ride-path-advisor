import pytest

from rpa.osm.gazetteer import Gazetteer, build_gazetteer, describe, read_places

OPL = """\
n1 x60.60 y56.84 Tplace=city,name=Екатеринбург
n2 x59.93 y56.80 Tplace=town,name=Ревда,population=62000
n3 x61.50 y57.90 Tplace=village,name=Ревда
n4 x60.30 y56.70 Tplace=village,name=Ключи
n5 x61.90 y56.20 Tplace=village,name=Ключи
n6 x59.50 y56.95 Trailway=halt,name=Дидинский%20%Тоннель
n7 x59.5001 y56.9501 Trailway=stop,name=Дидинский%20%Тоннель
n8 x60.10 y56.60 Tplace=hamlet,name=Берёзовка
n9 x60.70 y56.85 Tshop=supermarket,name=Шарташ
n10 x60.80 y56.90
n11 x60.82 y56.90
n12 x60.82 y56.91
n13 x60.80 y56.91
n14 x60.70 y56.86
n15 x60.71 y56.86
n16 x60.71 y56.87
n20 x59.00 y57.00
n21 x59.01 y57.00
n22 x59.02 y57.00
n23 x59.90 y57.20
n30 x60.20 y56.75 Tnatural=peak,name=Волчиха,alt_name=Большая%20%Волчиха;Волчья%20%гора
n31 x60.40 y56.50 Tplace=village,name=Озеро
n32 x60.41 y56.51 Tplace=hamlet
n33 x60.31 y56.71 Tplace=hamlet,name=Ключ
n34 x60.60 y56.85 Trailway=station,name=Круглое
n40 x59.95 y56.76
n41 x60.05 y56.76
n42 x60.05 y56.84
n43 x59.95 y56.84
n44 x60.805 y56.902
n45 x60.810 y56.902
n46 x60.810 y56.904
w1 Tnatural=water,water=lake,name=озеро%20%Круглое Nn10,n11,n12,n13,n10
w2 Tleisure=park,name=Шарташский%20%лесной%20%парк Nn14,n15,n16,n14
w3 Twaterway=river,name=Чусовая Nn20,n21
w4 Twaterway=river,name=Чусовая Nn21,n22
w5 Twaterway=river,name=Чусовая Nn22,n23
w6 Tnatural=water,water=wastewater,name=Отстойник Nn14,n15,n16,n14
w7 Thighway=residential,name=улица%20%Ревда Nn20,n21
w8 Tplace=town,name=Ревда Nn40,n41,n42,n43,n40
w9 Tnatural=water,water=lake,name=Круглое Nn44,n45,n46,n44
w10 Tnatural=water,water=river,name=Чусовая Nn20,n21,n22,n20
"""

EKB = (60.60, 56.84)


@pytest.fixture(scope="module")
def gazetteer(tmp_path_factory) -> Gazetteer:
    directory = tmp_path_factory.mktemp("gazetteer")
    (directory / "map.opl").write_text(OPL)
    build_gazetteer(read_places(directory / "map.opl"), directory / "gazetteer.sqlite")
    return Gazetteer(directory / "gazetteer.sqlite")


def names(hits) -> list[tuple[str, str]]:
    return [(hit.place.name, hit.place.kind) for hit in hits]


def test_town_outranks_a_village_of_the_same_name(gazetteer):
    hits = gazetteer.search("Ревда", near=(61.5, 57.9))  # standing in the village
    assert names(hits) == [("Ревда", "town"), ("Ревда", "village")]
    assert all(hit.exact for hit in hits)
    assert hits[1].distance_km < 1 < hits[0].distance_km
    # The town's node and its outline are one place: the node's position, the outline's size.
    assert hits[0].place.lon == 59.93 and hits[0].place.radius_m == pytest.approx(4150, rel=0.05)


def test_among_equals_the_nearer_comes_first(gazetteer):
    hits = [hit for hit in gazetteer.search("Ключи", near=EKB) if hit.exact]
    assert [round(hit.place.lon, 1) for hit in hits] == [60.3, 61.9]
    assert hits[0].distance_km == pytest.approx(24, abs=2)
    assert [round(hit.place.lon, 1) for hit in gazetteer.search("Ключи", near=(62.0, 56.2))][:2] == [61.9, 60.3]


def test_word_endings_do_not_matter_unless_a_name_matches_letter_for_letter(gazetteer):
    assert names(gazetteer.search("Ревду", near=EKB))[0] == ("Ревда", "town")
    assert gazetteer.search("до Екатеринбурга") == []
    assert names(gazetteer.search("Екатеринбурга")) == [("Екатеринбург", "city")]
    assert names(gazetteer.search("Круглый пруд"))[0] == ("озеро Круглое", "lake")
    # "Ключ" and "Ключи" differ only in the ending: whichever was typed is the exact one.
    assert [(hit.place.name, hit.exact) for hit in gazetteer.search("Ключ", near=EKB)][:2] == [
        ("Ключ", True),
        ("Ключи", False),
    ]
    assert [hit.place.name for hit in gazetteer.search("Ключи", near=EKB) if not hit.exact] == ["Ключ"]
    assert [hit.exact for hit in gazetteer.search("Ключа", near=EKB)] == [True, True, True]


def test_a_lake_outranks_the_station_named_after_it(gazetteer):
    assert names(gazetteer.search("Круглое", near=EKB)) == [("озеро Круглое", "lake"), ("Круглое", "station")]


def test_case_and_yo_do_not_matter(gazetteer):
    assert names(gazetteer.search("березовка", near=EKB)) == [("Берёзовка", "hamlet")]
    assert names(gazetteer.search("ЕКАТЕРИНБУРГ")) == [("Екатеринбург", "city")]


def test_kind_words_are_not_part_of_the_name(gazetteer):
    for query in ("озеро Круглое", "Круглое", "Круглое озеро", "оз. Круглое", "озера Круглого"):
        hit = gazetteer.search(query, near=EKB)[0]
        assert (hit.place.name, hit.place.kind, hit.exact) == ("озеро Круглое", "lake", True)
    assert names(gazetteer.search("город Ревда", near=EKB))[0] == ("Ревда", "town")


def test_a_kind_word_prefers_places_of_that_kind(gazetteer):
    assert names(gazetteer.search("деревня Ревда", near=EKB))[0] == ("Ревда", "village")


def test_a_name_made_of_a_kind_word_is_still_found(gazetteer):
    assert names(gazetteer.search("Озеро", near=EKB)) == [("Озеро", "village")]


def test_railway_halt_is_found_once(gazetteer):
    (hit,) = gazetteer.search("Дидинский тоннель", near=EKB)
    assert (hit.place.name, hit.place.kind, hit.exact) == ("Дидинский Тоннель", "station", True)
    assert names(gazetteer.search("станция Дидинский Тоннель")) == [("Дидинский Тоннель", "station")]


def test_prefix_matches_are_offered_as_inexact(gazetteer):
    hits = gazetteer.search("Шарташ", near=EKB)
    assert names(hits) == [("Шарташский лесной парк", "park")]  # the shop is not a place to ride to
    assert not hits[0].exact


def test_alternative_names_count(gazetteer):
    for query in ("Волчиха", "гора Волчиха", "Большая Волчиха", "Волчья гора"):
        (hit,) = gazetteer.search(query)
        assert (hit.place.name, hit.place.kind, hit.exact) == ("Волчиха", "peak", True)


def test_river_pieces_merge_within_a_cell_and_the_nearest_wins(gazetteer):
    hits = gazetteer.search("река Чусовая", near=(59.9, 57.2))
    assert {hit.place.kind for hit in hits} == {"river"}
    assert len(hits) == 2  # short pieces and a stretch mapped as an area side by side, one piece far away
    assert hits[0].distance_km < hits[1].distance_km


def test_areas_know_their_size(gazetteer):
    hit = gazetteer.search("Круглое")[0]  # the small namesake inside it is the same lake
    assert (hit.place.lon, hit.place.lat) == pytest.approx((60.81, 56.905), abs=0.006)
    assert hit.place.radius_m == pytest.approx(660, rel=0.1)  # about 1.2 km by 1.1 km


def test_unwanted_and_unknown_things_are_not_found(gazetteer):
    assert gazetteer.search("Отстойник") == []
    assert gazetteer.search("улица Ревда") == []  # streets are not indexed
    assert gazetteer.search("Нигденет") == []
    assert gazetteer.search("река") == []
    assert gazetteer.search("  ") == []


def test_limit_caps_the_answer(gazetteer):
    assert len(gazetteer.search("Ключи", near=EKB, limit=1)) == 1


def test_places_are_described_for_people(gazetteer):
    assert describe(gazetteer.search("Ревда")[0].place) == "Ревда (город)"
    assert describe(gazetteer.search("Круглое")[0].place) == "озеро Круглое"
    assert describe(gazetteer.search("Дидинский тоннель")[0].place) == "Дидинский Тоннель (станция)"
