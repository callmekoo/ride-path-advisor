"""Named places of the region, searchable by name: towns, stations, lakes, rivers, peaks, parks.

People name places that are not on the road graph at all (a railway halt, a
summit, the middle of a lake), so the gazetteer is read from the OSM file
itself; whoever uses a hit snaps it to the nearest road.
"""

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import osmium
import shapely

GAZETTEER_FILE = "gazetteer.sqlite"

_PLACE_KINDS = {
    "city": "city",
    "town": "town",
    "village": "village",
    "hamlet": "hamlet",
    "isolated_dwelling": "hamlet",
    "suburb": "suburb",
    "quarter": "suburb",
    "neighbourhood": "suburb",
    "locality": "locality",
    "island": "locality",
}
_SIGHTS = {"viewpoint", "attraction", "camp_site", "picnic_site", "museum"}
_NATURAL_SIGHTS = {"rock", "cliff", "cave_entrance", "spring", "stone"}
# Water nobody rides to (the same list as for routes "to the water").
_UNWANTED_WATER = {"wastewater", "sewage", "basin", "reflecting_pool", "moat", "drain", "ditch", "pool"}
_ALT_NAME_TAGS = ("alt_name", "old_name", "short_name", "official_name", "loc_name")

_WATER_KINDS = ("lake", "pond", "reservoir", "water")
_FLOWING_WATER = {"river", "stream", "canal", "oxbow"}

# Lower is more likely to be what a bare name means.
_RANK = {"city": 0, "town": 1, "village": 2, "station": 4, "river": 5, "locality": 5, "sight": 5}
_DEFAULT_RANK = 3

_KIND_WORD = {
    "city": "город",
    "town": "город",
    "village": "село",
    "hamlet": "деревня",
    "suburb": "район",
    "locality": "урочище",
    "station": "станция",
    "lake": "озеро",
    "pond": "пруд",
    "reservoir": "водохранилище",
    "water": "водоём",
    "river": "река",
    "peak": "гора",
    "park": "парк",
    "sight": "место",
}

# Words that say what a place is rather than what it is called. They are left out
# when names are compared; those with kinds also say which namesake is meant.
_GENERIC_WORDS: dict[str, tuple[str, ...]] = {
    "озеро": _WATER_KINDS,
    "оз": _WATER_KINDS,
    "пруд": _WATER_KINDS,
    "водохранилище": _WATER_KINDS,
    "вдхр": _WATER_KINDS,
    "река": ("river",),
    "речка": ("river",),
    "город": ("city", "town"),
    "поселок": ("village", "hamlet", "town"),
    "пос": ("village", "hamlet", "town"),
    "пгт": ("village", "town"),
    "село": ("village",),
    "деревня": ("hamlet", "village"),
    "дер": ("hamlet", "village"),
    "станция": ("station",),
    "ст": ("station",),
    "платформа": ("station",),
    "гора": ("peak",),
    "парк": ("park",),
    "лесопарк": ("park",),
    "тоннель": (),
    "туннель": (),
}


def _stem(token: str) -> str:
    """The word without its ending, so that cases and genders compare equal.

    Crude on purpose: "Ревду" and "Ревда", "Исетский" and "Исетское" must meet;
    telling apart every pair of similar names is left to the ranking.
    """
    stem = re.sub(r"(?:ого|его|ому|ему|ыми|ими|ами|ями|ым|им|ых|их|ом|ем|ам|ям|ах|ях)$", "", token)
    stem = re.sub(r"[аеийоуыьэюя]+$", "", stem)
    return stem if len(stem) >= 3 else token


_GENERIC = {_stem(word): kinds for word, kinds in _GENERIC_WORDS.items()}

# Namesakes of one kind this close together are one place (pieces of a river, a town and its outline).
_MERGE_DISTANCE_M = 4000.0
_M_PER_DEG = 111_195.0


@dataclass(frozen=True)
class Place:
    name: str
    kind: str  # a key of _KIND_WORD
    lon: float
    lat: float
    radius_m: float = 0.0  # rough size of an area; 0 for a point
    alt_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    place: Place
    exact: bool  # the whole name matches, not just the beginning of its words
    hinted: bool  # its kind is the one the query named ("озеро ..."); True when the query named none
    distance_km: float  # from the `near` point; 0 without one


def describe(place: Place) -> str:
    """The place as shown to a person: "Ревда (город)", "озеро Шарташ"."""
    says_what = any(_GENERIC.get(_stem(word)) for word in _tokens(place.name))
    return place.name if says_what else f"{place.name} ({_KIND_WORD[place.kind]})"


def rank(kind: str) -> int:
    """Importance class of a kind of place; lower is more important."""
    return _RANK.get(kind, _DEFAULT_RANK)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[0-9a-zа-я]+", text.lower().replace("ё", "е"))


@dataclass(frozen=True)
class _Name:
    key: str  # the name without kind words: sorted word stems
    plain: str  # the same words, unstemmed
    hinted: frozenset[str]  # kinds that the kind words point at
    only_kind_words: bool


def _name(text: str) -> _Name:
    tokens = _tokens(text)
    core = [token for token in tokens if _stem(token) not in _GENERIC]
    hinted = frozenset(kind for token in tokens for kind in _GENERIC.get(_stem(token), ()))
    if not core:
        # "Озеро" is a fine name for a village; a name that is nothing but kind words stays as it is.
        core, hinted = tokens, frozenset()
    return _Name(" ".join(sorted(map(_stem, core))), " ".join(sorted(core)), hinted, core is tokens)


def _distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    east = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(east, lat2 - lat1) * _M_PER_DEG


# --- reading OSM -------------------------------------------------------------


def _node_kind(tags: osmium.osm.TagList) -> str | None:
    if tags.get("place") in _PLACE_KINDS:
        return _PLACE_KINDS[tags["place"]]
    if tags.get("railway") in ("station", "halt"):
        return "station"
    if tags.get("natural") in ("peak", "volcano"):
        return "peak"
    if tags.get("natural") in _NATURAL_SIGHTS or tags.get("tourism") in _SIGHTS or "historic" in tags:
        return "sight"
    return None


def _area_kind(tags: osmium.osm.TagList) -> str | None:
    if tags.get("natural") == "water":
        water = tags.get("water", "")
        if water in _UNWANTED_WATER:
            return None
        if water in _FLOWING_WATER:
            return "river"
        return water if water in _WATER_KINDS else "water"
    if tags.get("leisure") in ("park", "nature_reserve") or tags.get("boundary") in ("national_park", "protected_area"):
        return "park"
    if tags.get("place") in _PLACE_KINDS:
        return _PLACE_KINDS[tags["place"]]
    return None


def _alt_names(tags: osmium.osm.TagList) -> tuple[str, ...]:
    found = [part.strip() for tag in _ALT_NAME_TAGS for part in tags.get(tag, "").split(";")]
    return tuple(name for name in found if name)


def read_places(path: Path) -> list[Place]:
    """Named places of an OSM file, with namesakes of one kind merged when they are close together."""
    factory = osmium.geom.WKBFactory()
    areas = osmium.filter.KeyFilter("natural", "leisure", "boundary", "place")
    processor = (
        osmium.FileProcessor(str(path))
        .with_locations()
        .with_areas(areas)
        .with_filter(osmium.filter.KeyFilter("name"))
    )
    found: dict[tuple[str, str], list[Place]] = {}
    for obj in processor:
        tags = obj.tags
        name = tags.get("name", "").strip()
        if not name:
            continue
        radius = 0.0
        try:
            if obj.is_node():
                kind = _node_kind(tags)
                if kind is None:
                    continue
                lon, lat = obj.location.lon, obj.location.lat
            elif obj.is_area():
                kind = _area_kind(tags)
                if kind is None:
                    continue
                shape = shapely.from_wkb(factory.create_multipolygon(obj))
                inside = shape.representative_point()
                lon, lat = inside.x, inside.y
                if kind != "river":  # a river mapped as an area is still a line to ride along, not a blob
                    area_m2 = shape.area * _M_PER_DEG**2 * math.cos(math.radians(lat))
                    radius = math.sqrt(area_m2 / math.pi)
            elif obj.is_way() and tags.get("waterway") == "river":
                kind = "river"
                middle = obj.nodes[len(obj.nodes) // 2].location
                lon, lat = middle.lon, middle.lat
            else:
                continue
        except (RuntimeError, osmium.InvalidLocationError):
            continue  # broken geometry, or nodes missing at the edge of the extract
        found.setdefault((_name(name).key, kind), []).append(Place(name, kind, lon, lat, radius, _alt_names(tags)))
    return [place for namesakes in found.values() for place in _merge_close(namesakes)]


def _merge_close(namesakes: list[Place]) -> list[Place]:
    """One place per cluster of namesakes: a town's node and its outline, the pieces of a river.

    Points win over areas (a place node marks the centre, an outline's inner
    point is arbitrary) and take over the area's size; among areas the largest
    wins: the lake rather than a bay mapped inside it.
    """
    kept: list[Place] = []
    for place in sorted(namesakes, key=lambda place: (place.radius_m > 0, -place.radius_m)):
        for i, other in enumerate(kept):
            reach = max(_MERGE_DISTANCE_M, 2 * place.radius_m, 2 * other.radius_m)
            if _distance_m(place.lon, place.lat, other.lon, other.lat) <= reach:
                if place.radius_m > other.radius_m and other.radius_m == 0:
                    kept[i] = Place(other.name, other.kind, other.lon, other.lat, place.radius_m, other.alt_names)
                break
        else:
            kept.append(place)
    return kept


# --- the index ---------------------------------------------------------------

_SCHEMA = """
CREATE TABLE places (
    id INTEGER PRIMARY KEY, name TEXT, kind TEXT, lon REAL, lat REAL, radius_m REAL, alt_names TEXT
);
CREATE TABLE keys (key TEXT, plain TEXT, place_id INTEGER);
CREATE INDEX keys_by_key ON keys(key);
CREATE VIRTUAL TABLE words USING fts5(text, tokenize = "unicode61 remove_diacritics 0");
"""


def build_gazetteer(places: list[Place], path: Path) -> None:
    """Write the search index, replacing any earlier one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript(_SCHEMA)
        for place_id, place in enumerate(places, start=1):
            connection.execute(
                "INSERT INTO places VALUES (?, ?, ?, ?, ?, ?, ?)",
                (place_id, place.name, place.kind, place.lon, place.lat, place.radius_m, ";".join(place.alt_names)),
            )
            all_names = (place.name, *place.alt_names)
            for name in {(_name(text).key, _name(text).plain) for text in all_names}:
                connection.execute("INSERT INTO keys VALUES (?, ?, ?)", (*name, place_id))
            text = " ".join(token for name in all_names for token in _tokens(name))
            connection.execute("INSERT INTO words (rowid, text) VALUES (?, ?)", (place_id, text))
    connection.close()


class Gazetteer:
    def __init__(self, path: Path) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self.path = Path(path)

    def __len__(self) -> int:
        with sqlite3.connect(self.path) as connection:
            return connection.execute("SELECT COUNT(*) FROM places").fetchone()[0]

    def search(self, query: str, near: tuple[float, float] | None = None, limit: int = 8) -> list[Hit]:
        """Places called `query`, most likely first.

        Whole-name matches come before matches on the beginnings of words;
        then places of the kind the query names ("озеро ..."), then the more
        important kinds, then the nearer to `near` (lon, lat).
        """
        name = _name(query)
        if not name.key:
            return []
        connection = sqlite3.connect(self.path)
        try:
            columns = "p.id, p.name, p.kind, p.lon, p.lat, p.radius_m, p.alt_names"
            matching = connection.execute(
                f"SELECT {columns}, MAX(k.plain = ?) FROM keys k JOIN places p ON p.id = k.place_id "
                "WHERE k.key = ? GROUP BY p.id",
                (name.plain, name.key),
            ).fetchall()
            # A bare kind word ("река") can only be a place's whole name, never the beginning of one.
            loose = (
                []
                if name.only_kind_words
                else connection.execute(
                    f"SELECT {columns} FROM words w JOIN places p ON p.id = w.rowid WHERE words MATCH ? LIMIT 200",
                    (" ".join(f'"{stem}"*' for stem in name.key.split()),),
                ).fetchall()
            )
        finally:
            connection.close()

        # Names equal up to word endings are exact, unless some are equal outright: then only those are.
        letter_for_letter = any(row[7] for row in matching)
        exact_ids = {row[0] for row in matching if row[7] or not letter_for_letter}
        rows = [(row[:7], row[0] in exact_ids) for row in matching]
        rows += [(row, False) for row in loose if row[0] not in {row[0] for row in matching}]
        hits = [self._hit(row, exact, name.hinted, near) for row, exact in rows]
        hits.sort(key=lambda hit: (not hit.exact, not hit.hinted, rank(hit.place.kind), hit.distance_km))
        return hits[:limit]

    @staticmethod
    def _hit(row: tuple, exact: bool, hinted: frozenset[str], near: tuple[float, float] | None) -> Hit:
        _, name, kind, lon, lat, radius, alt_names = row
        distance = _distance_m(*near, lon, lat) / 1000 if near is not None else 0.0
        place = Place(name, kind, lon, lat, radius, tuple(filter(None, alt_names.split(";"))))
        return Hit(place, exact, not hinted or kind in hinted, distance)

