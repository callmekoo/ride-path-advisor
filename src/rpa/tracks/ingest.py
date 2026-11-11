"""Import rides from the inbox: Strava export archives and loose track files."""

import csv
import io
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rpa.tracks.clean import Bbox, clean, step_m
from rpa.tracks.parse import kind_of, parse_track
from rpa.tracks.store import TrackSet, make_trackset

# Columns of Strava's activities.csv. The header is translated into the
# account's language, so columns are addressed by position.
_COL_ID, _COL_TYPE, _COL_COMMUTE, _COL_GEAR, _COL_FILE = 0, 3, 9, 11, 12

# Activity types that are outdoor bicycle rides, as Strava names them in
# English and Russian exports. Anything else is skipped and reported.
RIDE_TYPES = {
    "Ride",
    "Gravel Ride",
    "Mountain Bike Ride",
    "E-Bike Ride",
    "E-Mountain Bike Ride",
    "Велосипед",
    "Велозаезд",
    "Гравийный велозаезд",
    "Заезд на горном велосипеде",
    "Заезд на электровелосипеде",
}


@dataclass
class Job:
    """One track file to load, with what is known about it."""

    meta: dict
    data: bytes
    kind: str
    gzipped: bool


@dataclass
class ImportReport:
    files: int = 0
    imported: int = 0
    km: float = 0.0
    skipped_types: Counter = field(default_factory=Counter)
    outside_region: int = 0
    unreadable: list[str] = field(default_factory=list)


def _strava_jobs(path: Path, report: ImportReport) -> Iterator[Job]:
    with zipfile.ZipFile(path) as archive:
        with archive.open("activities.csv") as file:
            rows = list(csv.reader(io.TextIOWrapper(file, "utf-8-sig")))[1:]
        for row in rows:
            activity_type, filename = row[_COL_TYPE], row[_COL_FILE]
            if activity_type not in RIDE_TYPES:
                report.skipped_types[activity_type] += 1
                continue
            kind = kind_of(filename)
            if kind is None:  # an activity entered by hand has no file
                continue
            meta = {
                "id": f"strava:{row[_COL_ID]}",
                "source": path.name,
                "type": activity_type,
                "gear": row[_COL_GEAR],
                "commute": row[_COL_COMMUTE] == "true",
            }
            yield Job(meta, archive.read(filename), *kind)


def _inbox_jobs(inbox: Path, report: ImportReport) -> Iterator[Job]:
    for path in sorted(Path(inbox).iterdir()):
        if path.name.startswith("."):
            continue  # a file still being uploaded, or the system's own
        if path.suffix.lower() == ".zip":
            yield from _strava_jobs(path, report)
        elif (kind := kind_of(path.name)) is not None:
            meta = {"id": f"file:{path.name}", "source": path.name, "type": "", "gear": "", "commute": False}
            yield Job(meta, path.read_bytes(), *kind)


def _load(job: Job, bbox: Bbox) -> tuple[list[np.ndarray], str | None, str | None]:
    """(segments, ISO date of the first point, error) for one file. Runs in a worker process."""
    try:
        raw = parse_track(job.data, job.kind, job.gzipped)
    except Exception as error:  # a corrupt file must not stop the import
        return [], None, f"{job.meta['id']}: {type(error).__name__}: {error}"
    times = raw.time[np.isfinite(raw.time)]
    date = datetime.fromtimestamp(float(times[0]), UTC).date().isoformat() if len(times) else None
    return clean(raw.lonlat, bbox), date, None


def import_inbox(inbox: Path, bbox: Bbox, log: Callable[[str], None] = print) -> tuple[TrackSet, ImportReport]:
    """Load every ride found in `inbox`, keeping the parts that lie inside `bbox`."""
    report = ImportReport()
    seen: set[str] = set()
    jobs = []
    for job in _inbox_jobs(inbox, report):
        if job.meta["id"] not in seen:  # the same activity may sit in two archives
            seen.add(job.meta["id"])
            jobs.append(job)
    report.files = len(jobs)
    log(f"Файлов треков: {len(jobs)}")

    tracks, segments = [], []
    with ProcessPoolExecutor() as pool:
        results = pool.map(_load, jobs, [bbox] * len(jobs), chunksize=16)
        for job, (pieces, date, error) in zip(jobs, results):
            if error:
                report.unreadable.append(error)
            elif not pieces:
                report.outside_region += 1
            else:
                tracks.append({**job.meta, "date": date})
                segments.append(pieces)
                report.km += sum(float(step_m(piece).sum()) for piece in pieces) / 1000
    report.imported = len(tracks)
    return make_trackset(tracks, segments), report
