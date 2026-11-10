"""Turn the tags of an OSM way into compact attributes for bicycle routing.

`classify_way` is a pure function over a tag mapping, so every rule here is
covered by a table in `tests/test_tags.py`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum


class Highway(IntEnum):
    CYCLEWAY = 0
    UNCLASSIFIED = 1
    TRACK = 2
    RESIDENTIAL = 3
    LIVING_STREET = 4
    SERVICE = 5
    TERTIARY = 6
    PATH = 7
    PEDESTRIAN = 8
    FOOTWAY = 9
    SECONDARY = 10
    PRIMARY = 11
    TRUNK = 12


class Surface(IntEnum):
    """How confident we are that the way is paved. `surface` is tagged on only
    a minority of rural ways, hence the in-between classes."""

    PAVED = 0
    LIKELY_PAVED = 1
    UNKNOWN = 2
    LIKELY_UNPAVED = 3
    UNPAVED = 4
    BAD = 5  # barely rideable: sand, mud, grade4-5 tracks


# Edge flags.
FWD_OK = 1  # may be ridden in the direction of the OSM way
BACK_OK = 2  # may be ridden against it
SOFT_RESTRICTED = 4  # legally doubtful but physically rideable; penalised, not banned
FORD = 8
BRIDGE = 16
TUNNEL = 32
CYCLE_INFRA = 64


@dataclass(frozen=True)
class EdgeAttrs:
    highway: Highway
    surface: Surface
    flags: int


_HIGHWAY = {
    "cycleway": Highway.CYCLEWAY,
    "unclassified": Highway.UNCLASSIFIED,
    "road": Highway.UNCLASSIFIED,
    "track": Highway.TRACK,
    "residential": Highway.RESIDENTIAL,
    "living_street": Highway.LIVING_STREET,
    "service": Highway.SERVICE,
    "tertiary": Highway.TERTIARY,
    "path": Highway.PATH,
    "bridleway": Highway.PATH,
    "pedestrian": Highway.PEDESTRIAN,
    "footway": Highway.FOOTWAY,
    "secondary": Highway.SECONDARY,
    "primary": Highway.PRIMARY,
    "trunk": Highway.TRUNK,
}

_SURFACE_TAG = {
    **dict.fromkeys(
        (
            "asphalt",
            "concrete",
            "concrete:plates",
            "concrete:lanes",
            "paved",
            "paving_stones",
            "sett",
            "chipseal",
            "bricks",
            "metal",
        ),
        Surface.PAVED,
    ),
    "wood": Surface.LIKELY_PAVED,
    **dict.fromkeys(
        (
            "unpaved",
            "compacted",
            "fine_gravel",
            "gravel",
            "pebblestone",
            "ground",
            "dirt",
            "earth",
            "grass_paver",
            "cobblestone",
            "unhewn_cobblestone",
        ),
        Surface.UNPAVED,
    ),
    **dict.fromkeys(("sand", "mud", "grass", "snow", "ice", "woodchips", "rock"), Surface.BAD),
}

_TRACKTYPE = {
    "grade1": Surface.LIKELY_PAVED,
    "grade2": Surface.UNPAVED,
    "grade3": Surface.UNPAVED,
    "grade4": Surface.BAD,
    "grade5": Surface.BAD,
}

_SURFACE_BY_HIGHWAY = {
    Highway.TRUNK: Surface.PAVED,
    Highway.PRIMARY: Surface.PAVED,
    Highway.SECONDARY: Surface.LIKELY_PAVED,
    Highway.TERTIARY: Surface.LIKELY_PAVED,
    Highway.CYCLEWAY: Surface.LIKELY_PAVED,
    Highway.TRACK: Surface.LIKELY_UNPAVED,
    Highway.PATH: Surface.LIKELY_UNPAVED,
}

_UNRIDEABLE_SMOOTHNESS = {"horrible", "very_horrible", "impassable"}
_BICYCLE_PERMITTED = {"yes", "designated", "permissive"}
_BICYCLE_DISCOURAGED = {"no", "use_sidepath", "dismount"}
_ACCESS_CLOSED = {"no", "military"}
_ACCESS_SOFT = {"private", "destination", "delivery", "customers"}
_VEHICLE_SOFT = {"no", "private"}
_ONEWAY_FORWARD = {"yes", "1", "true"}
_ONEWAY_BACKWARD = {"-1", "reverse"}
_ONEWAY_IGNORED_ON = {Highway.FOOTWAY, Highway.PATH, Highway.PEDESTRIAN}
_CYCLEWAY_KEYS = ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")
_CYCLEWAY_INFRA = {"lane", "track", "shared_lane", "share_busway", "opposite_lane", "opposite_track"}


def classify_way(tags: Mapping[str, str]) -> EdgeAttrs | None:
    """Return routing attributes of a way, or None if a bicycle cannot use it."""
    highway = _HIGHWAY.get(tags.get("highway", "").removesuffix("_link"))
    if highway is None:
        return None
    # Never rideable, whatever else is tagged. Winter roads cross swamps and
    # lakes on ice and do not exist in summer.
    if tags.get("motorroad") == "yes" or tags.get("winter_road") == "yes" or tags.get("ice_road") == "yes":
        return None

    bicycle = tags.get("bicycle")
    permitted = bicycle in _BICYCLE_PERMITTED
    access = tags.get("access")
    if access in _ACCESS_CLOSED and not permitted:
        return None

    flags = _direction_flags(tags, highway)
    soft = not permitted and (
        access in _ACCESS_SOFT
        or tags.get("vehicle") in _VEHICLE_SOFT
        or bicycle in _BICYCLE_DISCOURAGED
        or highway == Highway.FOOTWAY
    )
    if soft:
        flags |= SOFT_RESTRICTED
    if tags.get("ford") == "yes":
        flags |= FORD
    if tags.get("bridge", "no") != "no":
        flags |= BRIDGE
    if tags.get("tunnel", "no") != "no":
        flags |= TUNNEL
    if (
        highway == Highway.CYCLEWAY
        or bicycle == "designated"
        or any(tags.get(key) in _CYCLEWAY_INFRA for key in _CYCLEWAY_KEYS)
    ):
        flags |= CYCLE_INFRA

    return EdgeAttrs(highway, _surface(tags, highway), flags)


def _direction_flags(tags: Mapping[str, str], highway: Highway) -> int:
    contraflow_allowed = (
        highway in _ONEWAY_IGNORED_ON
        or tags.get("oneway:bicycle") == "no"
        or any(tags.get(key, "").startswith("opposite") for key in _CYCLEWAY_KEYS)
    )
    if contraflow_allowed:
        return FWD_OK | BACK_OK
    oneway = tags.get("oneway")
    if oneway in _ONEWAY_FORWARD:
        return FWD_OK
    if oneway in _ONEWAY_BACKWARD:
        return BACK_OK
    if tags.get("junction") in ("roundabout", "circular") and oneway != "no":
        return FWD_OK
    return FWD_OK | BACK_OK


def _surface(tags: Mapping[str, str], highway: Highway) -> Surface:
    if tags.get("smoothness") in _UNRIDEABLE_SMOOTHNESS:
        return Surface.BAD
    tagged = _SURFACE_TAG.get(tags.get("surface", ""))
    if tagged is not None:
        return tagged
    by_tracktype = _TRACKTYPE.get(tags.get("tracktype", ""))
    if by_tracktype is not None:
        return by_tracktype
    return _SURFACE_BY_HIGHWAY.get(highway, Surface.UNKNOWN)
