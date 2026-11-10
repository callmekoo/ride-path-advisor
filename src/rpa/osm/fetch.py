"""Download regional OSM extracts and cut out the area the service covers.

The area is the boundary polygon of the home region grown by a buffer, so that
rides just across the administrative border still have roads under them. The
resulting `region.pbf` is the single source for both our graph and the
map-matching tiles.

Merging and clipping are done by osmium-tool: the system one if installed,
otherwise one inside a small podman image built on first use.
"""

import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.geometry.base import BaseGeometry

KM_PER_DEGREE = 111.195

OSMIUM_IMAGE = "rpa-osmium"
_OSMIUM_CONTAINERFILE = """\
FROM registry.fedoraproject.org/fedora-minimal:44
RUN microdnf -y install osmium-tool && microdnf clean all
WORKDIR /work
ENTRYPOINT ["osmium"]
"""


def parse_poly(text: str) -> BaseGeometry:
    """Parse the Osmosis `.poly` format: rings of "lon lat" lines; `!name` marks a hole."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    outers: list[list[tuple[float, float]]] = []
    holes: list[list[tuple[float, float]]] = []
    i = 1  # line 0 is the name of the file
    while i < len(lines) and lines[i] != "END":
        is_hole = lines[i].startswith("!")
        ring = []
        i += 1
        while lines[i] != "END":
            lon, lat = lines[i].split()[:2]
            ring.append((float(lon), float(lat)))
            i += 1
        i += 1
        (holes if is_hole else outers).append(ring)
    if not outers:
        raise ValueError("no polygon rings found in .poly text")
    shape = shapely.union_all([Polygon(ring) for ring in outers])
    for ring in holes:
        shape = shape.difference(Polygon(ring))
    return shape


def buffer_km(shape: BaseGeometry, km: float) -> BaseGeometry:
    """Grow a lon/lat shape by `km` kilometres.

    Works in a local flat projection centred on the shape, which is accurate
    to a few percent over a region a few hundred kilometres across. That is
    plenty for a clipping boundary.
    """
    lat0 = (shape.bounds[1] + shape.bounds[3]) / 2
    scale = np.array([KM_PER_DEGREE * np.cos(np.radians(lat0)), KM_PER_DEGREE])
    flat = shapely.transform(shape, lambda lonlat: lonlat * scale)
    grown = flat.buffer(km).simplify(1.0)
    return shapely.transform(grown, lambda xy: xy / scale)


def write_geojson(shape: BaseGeometry, path: Path) -> None:
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(shape)}
    Path(path).write_text(json.dumps(feature))


def osmium_command(args: list[str], workdir: Path, native: str | None) -> list[str]:
    """Command line that runs `osmium <args>` with `workdir` as its working directory."""
    if native:
        return [native, *args]
    volume = f"{Path(workdir).resolve()}:/work:Z"
    return ["podman", "run", "--rm", "--userns=keep-id", "-v", volume, OSMIUM_IMAGE, *args]


def _ensure_osmium_image() -> None:
    present = subprocess.run(["podman", "image", "exists", OSMIUM_IMAGE]).returncode == 0
    if not present:
        subprocess.run(
            ["podman", "build", "-t", OSMIUM_IMAGE, "-"], input=_OSMIUM_CONTAINERFILE, text=True, check=True
        )


def run_osmium(args: list[str], workdir: Path) -> None:
    """Run osmium-tool in `workdir`; file arguments must be relative to it."""
    native = shutil.which("osmium")
    if native is None:
        if shutil.which("podman") is None:
            raise RuntimeError("osmium-tool is needed: install it, or install podman to run it in a container")
        _ensure_osmium_image()
    subprocess.run(osmium_command(args, workdir, native), cwd=workdir, check=True)


def download(url: str, dest: Path, client: httpx.Client) -> None:
    """Download `url` to `dest` unless it is already there."""
    if dest.exists():
        return
    part = dest.with_name(dest.name + ".part")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with part.open("wb") as file:
            for chunk in response.iter_bytes(1 << 20):
                file.write(chunk)
    part.rename(dest)


def fetch_region(
    extract_urls: list[str],
    boundary_poly_url: str,
    buffer: float,
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> Path:
    """Produce `out_dir/region.pbf`: the extracts merged and clipped to boundary + buffer km."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        names = []
        for url in extract_urls:
            name = url.rsplit("/", 1)[-1]
            log(f"Скачиваю {name}")
            download(url, out_dir / name, client)
            names.append(name)
        boundary = client.get(boundary_poly_url)
        boundary.raise_for_status()

    write_geojson(buffer_km(parse_poly(boundary.text), buffer), out_dir / "region.geojson")
    log("Склеиваю выгрузки")
    run_osmium(["merge", *names, "-o", "merged.osm.pbf", "--overwrite"], out_dir)
    # Neighbouring extracts are snapshots taken minutes apart, so an object on
    # a border can arrive in two versions. `merge` keeps both and `extract`
    # refuses such input; `time-filter` keeps the latest version of each.
    run_osmium(["time-filter", "merged.osm.pbf", "-o", "latest.osm.pbf", "--overwrite"], out_dir)
    log(f"Вырезаю область с буфером {buffer:g} км")
    run_osmium(
        ["extract", "-p", "region.geojson", "latest.osm.pbf", "-o", "region.osm.pbf", "--overwrite"], out_dir
    )
    (out_dir / "merged.osm.pbf").unlink()
    (out_dir / "latest.osm.pbf").unlink()
    return out_dir / "region.osm.pbf"
