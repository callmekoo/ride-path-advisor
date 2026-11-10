import pytest

from rpa.osm.tags import (
    BACK_OK,
    BRIDGE,
    CYCLE_INFRA,
    FORD,
    FWD_OK,
    SOFT_RESTRICTED,
    TUNNEL,
    Highway,
    Surface,
    classify_way,
)

BOTH = FWD_OK | BACK_OK

NOT_ROUTABLE = [
    {},
    {"building": "yes"},
    {"highway": "motorway"},
    {"highway": "motorway_link"},
    {"highway": "construction"},
    {"highway": "proposed"},
    {"highway": "steps"},
    {"highway": "platform"},
    {"highway": "trunk", "motorroad": "yes"},
    {"highway": "track", "winter_road": "yes"},
    {"highway": "unclassified", "ice_road": "yes"},
    {"highway": "service", "access": "no"},
    {"highway": "track", "access": "military"},
    {"highway": "track", "winter_road": "yes", "bicycle": "yes"},
]


@pytest.mark.parametrize("tags", NOT_ROUTABLE)
def test_not_routable(tags):
    assert classify_way(tags) is None


HIGHWAY_CASES = [
    ("cycleway", Highway.CYCLEWAY),
    ("unclassified", Highway.UNCLASSIFIED),
    ("road", Highway.UNCLASSIFIED),
    ("track", Highway.TRACK),
    ("residential", Highway.RESIDENTIAL),
    ("living_street", Highway.LIVING_STREET),
    ("service", Highway.SERVICE),
    ("tertiary", Highway.TERTIARY),
    ("tertiary_link", Highway.TERTIARY),
    ("path", Highway.PATH),
    ("bridleway", Highway.PATH),
    ("pedestrian", Highway.PEDESTRIAN),
    ("footway", Highway.FOOTWAY),
    ("secondary", Highway.SECONDARY),
    ("secondary_link", Highway.SECONDARY),
    ("primary", Highway.PRIMARY),
    ("primary_link", Highway.PRIMARY),
    ("trunk", Highway.TRUNK),
    ("trunk_link", Highway.TRUNK),
]


@pytest.mark.parametrize(("value", "expected"), HIGHWAY_CASES)
def test_highway_class(value, expected):
    assert classify_way({"highway": value}).highway == expected


SURFACE_CASES = [
    # explicit surface tag
    ({"highway": "track", "surface": "asphalt"}, Surface.PAVED),
    ({"highway": "track", "surface": "concrete:plates"}, Surface.PAVED),
    ({"highway": "residential", "surface": "paving_stones"}, Surface.PAVED),
    ({"highway": "tertiary", "surface": "gravel"}, Surface.UNPAVED),
    ({"highway": "primary", "surface": "ground"}, Surface.UNPAVED),
    ({"highway": "residential", "surface": "compacted"}, Surface.UNPAVED),
    ({"highway": "track", "surface": "sand"}, Surface.BAD),
    ({"highway": "path", "surface": "mud"}, Surface.BAD),
    # smoothness can only make things worse
    ({"highway": "tertiary", "surface": "asphalt", "smoothness": "horrible"}, Surface.BAD),
    ({"highway": "track", "smoothness": "impassable"}, Surface.BAD),
    ({"highway": "tertiary", "surface": "asphalt", "smoothness": "bad"}, Surface.PAVED),
    # tracktype when surface is missing or unknown
    ({"highway": "track", "tracktype": "grade1"}, Surface.LIKELY_PAVED),
    ({"highway": "track", "tracktype": "grade2"}, Surface.UNPAVED),
    ({"highway": "track", "tracktype": "grade3"}, Surface.UNPAVED),
    ({"highway": "track", "tracktype": "grade4"}, Surface.BAD),
    ({"highway": "track", "tracktype": "grade5"}, Surface.BAD),
    ({"highway": "track", "surface": "something_new", "tracktype": "grade2"}, Surface.UNPAVED),
    ({"highway": "track", "surface": "asphalt", "tracktype": "grade5"}, Surface.PAVED),
    # highway class as the last resort
    ({"highway": "trunk"}, Surface.PAVED),
    ({"highway": "primary"}, Surface.PAVED),
    ({"highway": "secondary"}, Surface.LIKELY_PAVED),
    ({"highway": "tertiary"}, Surface.LIKELY_PAVED),
    ({"highway": "cycleway"}, Surface.LIKELY_PAVED),
    ({"highway": "unclassified"}, Surface.UNKNOWN),
    ({"highway": "residential"}, Surface.UNKNOWN),
    ({"highway": "service"}, Surface.UNKNOWN),
    ({"highway": "footway"}, Surface.UNKNOWN),
    ({"highway": "track"}, Surface.LIKELY_UNPAVED),
    ({"highway": "path"}, Surface.LIKELY_UNPAVED),
]


@pytest.mark.parametrize(("tags", "expected"), SURFACE_CASES)
def test_surface(tags, expected):
    assert classify_way(tags).surface == expected


DIRECTION_CASES = [
    ({"highway": "residential"}, BOTH),
    ({"highway": "residential", "oneway": "no"}, BOTH),
    ({"highway": "residential", "oneway": "yes"}, FWD_OK),
    ({"highway": "residential", "oneway": "1"}, FWD_OK),
    ({"highway": "residential", "oneway": "true"}, FWD_OK),
    ({"highway": "residential", "oneway": "-1"}, BACK_OK),
    ({"highway": "residential", "oneway": "reverse"}, BACK_OK),
    ({"highway": "tertiary", "junction": "roundabout"}, FWD_OK),
    ({"highway": "tertiary", "junction": "circular"}, FWD_OK),
    ({"highway": "tertiary", "junction": "roundabout", "oneway": "no"}, BOTH),
    ({"highway": "residential", "oneway": "yes", "oneway:bicycle": "no"}, BOTH),
    ({"highway": "residential", "oneway": "yes", "cycleway": "opposite"}, BOTH),
    ({"highway": "residential", "oneway": "yes", "cycleway:left": "opposite_lane"}, BOTH),
    ({"highway": "path", "oneway": "yes"}, BOTH),
    ({"highway": "pedestrian", "oneway": "yes"}, BOTH),
    ({"highway": "footway", "oneway": "yes", "bicycle": "yes"}, BOTH),
]


@pytest.mark.parametrize(("tags", "expected"), DIRECTION_CASES)
def test_direction(tags, expected):
    assert classify_way(tags).flags & BOTH == expected


SOFT_CASES = [
    ({"highway": "service", "access": "private"}, True),
    ({"highway": "service", "access": "destination"}, True),
    ({"highway": "service", "access": "customers"}, True),
    ({"highway": "service", "access": "delivery"}, True),
    ({"highway": "track", "vehicle": "no"}, True),
    ({"highway": "track", "vehicle": "private"}, True),
    ({"highway": "residential", "bicycle": "no"}, True),
    ({"highway": "primary", "bicycle": "use_sidepath"}, True),
    ({"highway": "pedestrian", "bicycle": "dismount"}, True),
    ({"highway": "footway"}, True),
    ({"highway": "footway", "bicycle": "yes"}, False),
    ({"highway": "footway", "bicycle": "designated"}, False),
    ({"highway": "service", "access": "private", "bicycle": "permissive"}, False),
    ({"highway": "track", "vehicle": "no", "bicycle": "yes"}, False),
    ({"highway": "track", "motor_vehicle": "no"}, False),
    ({"highway": "residential"}, False),
    ({"highway": "path"}, False),
]


@pytest.mark.parametrize(("tags", "expected"), SOFT_CASES)
def test_soft_restriction(tags, expected):
    assert bool(classify_way(tags).flags & SOFT_RESTRICTED) == expected


def test_explicit_bicycle_permission_overrides_access_no():
    attrs = classify_way({"highway": "service", "access": "no", "bicycle": "yes"})
    assert attrs is not None
    assert not attrs.flags & SOFT_RESTRICTED


FEATURE_CASES = [
    ({"highway": "track", "ford": "yes"}, FORD),
    ({"highway": "tertiary", "bridge": "yes"}, BRIDGE),
    ({"highway": "tertiary", "bridge": "viaduct"}, BRIDGE),
    ({"highway": "tertiary", "tunnel": "yes"}, TUNNEL),
    ({"highway": "cycleway"}, CYCLE_INFRA),
    ({"highway": "path", "bicycle": "designated"}, CYCLE_INFRA),
    ({"highway": "secondary", "cycleway": "lane"}, CYCLE_INFRA),
    ({"highway": "secondary", "cycleway:right": "track"}, CYCLE_INFRA),
    ({"highway": "secondary", "cycleway:both": "shared_lane"}, CYCLE_INFRA),
]


@pytest.mark.parametrize(("tags", "flag"), FEATURE_CASES)
def test_feature_flag_set(tags, flag):
    assert classify_way(tags).flags & flag


@pytest.mark.parametrize(
    "tags",
    [
        {"highway": "tertiary"},
        {"highway": "tertiary", "bridge": "no", "tunnel": "no", "ford": "no"},
        {"highway": "secondary", "cycleway": "no"},
    ],
)
def test_feature_flags_absent(tags):
    assert not classify_way(tags).flags & (FORD | BRIDGE | TUNNEL | CYCLE_INFRA)
