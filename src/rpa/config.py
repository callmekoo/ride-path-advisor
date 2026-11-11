"""Where things live on disk. Everything personal stays under the data directory."""

import os
from pathlib import Path


def data_dir() -> Path:
    """Root for downloaded and derived data; override with RPA_DATA_DIR."""
    return Path(os.environ.get("RPA_DATA_DIR", "data"))


def graph_root() -> Path:
    return data_dir() / "graph"


def current_graph_dir() -> Path:
    """Symlink to the graph that commands use by default."""
    return graph_root() / "current"


def osm_dir() -> Path:
    return data_dir() / "osm"


# The covered area: the home region grown by a buffer, so that rides just
# across the administrative border still have roads under them. Neighbouring
# extracts supply the data for the buffer.
_EXTRACTS = "https://download.openstreetmap.fr/extracts/russia/"
REGION_EXTRACT_URLS = [
    _EXTRACTS + "ural_federal_district/sverdlovsk_oblast-latest.osm.pbf",
    _EXTRACTS + "ural_federal_district/chelyabinsk_oblast-latest.osm.pbf",
    _EXTRACTS + "ural_federal_district/kurgan_oblast-latest.osm.pbf",
    _EXTRACTS + "ural_federal_district/tyumen_oblast-latest.osm.pbf",
    _EXTRACTS + "volga_federal_district/perm_krai-latest.osm.pbf",
    _EXTRACTS + "volga_federal_district/bashkortostan_republic-latest.osm.pbf",
]
REGION_BOUNDARY_POLY_URL = (
    "https://download.openstreetmap.fr/polygons/russia/ural_federal_district/sverdlovsk_oblast.poly"
)
REGION_BUFFER_KM = 100.0


def inbox_dir() -> Path:
    """Drop Strava export archives and loose GPX/FIT/TCX files here."""
    return data_dir() / "inbox"


def tracks_dir() -> Path:
    return data_dir() / "tracks"


def valhalla_dir() -> Path:
    return data_dir() / "valhalla"


def match_file() -> Path:
    """Map-matched tracks, independent of the graph version."""
    return data_dir() / "match" / "matched.npz"


def region_pbf() -> Path:
    return osm_dir() / "region.osm.pbf"


def db_file() -> Path:
    """Saved places and recent route requests."""
    return data_dir() / "rpa.sqlite"
