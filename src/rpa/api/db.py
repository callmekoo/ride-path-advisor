"""What the service remembers between requests: saved places, recent route requests, saved routes.

One small SQLite file. Recent requests are kept so that a GPX link still works after
a restart and so that a follow-up ("shorter", "not through that town") can see what
was asked and offered before. A route the rider saves is copied out of them, so it
stays when its request is dropped.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

KEEP_REQUESTS = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS places (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_key TEXT NOT NULL UNIQUE,  -- casefolded: SQLite's NOCASE only folds ASCII
    lon REAL NOT NULL,
    lat REAL NOT NULL,
    is_default INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    spec TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS candidates_by_request ON candidates(request_id);
CREATE TABLE IF NOT EXISTS routes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    name TEXT NOT NULL,
    spec TEXT NOT NULL,
    body TEXT NOT NULL  -- the candidate as JSON
);
"""

_PLACE_COLUMNS = "id, name, lon, lat, is_default"
_ROUTE_COLUMNS = "id, name, created, json_extract(body, '$.length_km'), json_extract(body, '$.ascent_m')"


class DuplicateName(Exception):
    """A place with this name already exists (names compare case-insensitively)."""


def _place(row: tuple) -> dict:
    return {"id": row[0], "name": row[1], "lon": row[2], "lat": row[3], "is_default": bool(row[4])}


def _route(row: tuple) -> dict:
    return {"id": row[0], "name": row[1], "created": row[2], "length_km": row[3], "ascent_m": row[4]}


class Db:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection for one unit of work: committed on success, rolled back on error."""
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    # Places

    def places(self) -> list[dict]:
        with self._connect() as connection:
            return [_place(row) for row in connection.execute(f"SELECT {_PLACE_COLUMNS} FROM places ORDER BY id")]

    def default_place(self) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(f"SELECT {_PLACE_COLUMNS} FROM places WHERE is_default").fetchone()
        return _place(row) if row else None

    def add_place(self, name: str, lon: float, lat: float, is_default: bool = False) -> dict:
        with self._connect() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO places (name, name_key, lon, lat) VALUES (?, ?, ?, ?)", (name, name.casefold(), lon, lat)
                )
            except sqlite3.IntegrityError:
                raise DuplicateName(name) from None
            if is_default:
                _make_default(connection, cursor.lastrowid)
            _ensure_default(connection)
            return _read_place(connection, cursor.lastrowid)

    def update_place(
        self,
        place_id: int,
        name: str | None = None,
        lon: float | None = None,
        lat: float | None = None,
        is_default: bool | None = None,
    ) -> dict | None:
        """Change the given fields; None if there is no such place.

        Clearing `is_default` has no effect: the default moves only by being given to another place.
        """
        with self._connect() as connection:
            if _read_place(connection, place_id) is None:
                return None
            changes = {"name": name, "name_key": name and name.casefold(), "lon": lon, "lat": lat}
            changes = {column: value for column, value in changes.items() if value is not None}
            if changes:
                assignments = ", ".join(f"{column} = ?" for column in changes)
                try:
                    connection.execute(f"UPDATE places SET {assignments} WHERE id = ?", (*changes.values(), place_id))
                except sqlite3.IntegrityError:
                    raise DuplicateName(name) from None
            if is_default:
                _make_default(connection, place_id)
            return _read_place(connection, place_id)

    def delete_place(self, place_id: int) -> bool:
        with self._connect() as connection:
            deleted = connection.execute("DELETE FROM places WHERE id = ?", (place_id,)).rowcount > 0
            _ensure_default(connection)
            return deleted

    # Requests and the candidates offered for them

    def save_request(self, spec: str, candidates: list[str]) -> tuple[int, list[int]]:
        """Store a request with its candidates; returns the request id and the candidate ids in order."""
        with self._connect() as connection:
            request_id = connection.execute("INSERT INTO requests (spec) VALUES (?)", (spec,)).lastrowid
            ids = [
                connection.execute(
                    "INSERT INTO candidates (request_id, body) VALUES (?, ?)", (request_id, body)
                ).lastrowid
                for body in candidates
            ]
            connection.execute("DELETE FROM requests WHERE id <= ?", (request_id - KEEP_REQUESTS,))
            return request_id, ids

    def candidate(self, candidate_id: int) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT body FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        return row[0] if row else None

    def request(self, request_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute("SELECT spec FROM requests WHERE id = ?", (request_id,)).fetchone()
            if row is None:
                return None
            bodies = connection.execute("SELECT body FROM candidates WHERE request_id = ? ORDER BY id", (request_id,))
            return {"spec": row[0], "candidates": [body for (body,) in bodies]}

    # Saved routes

    def save_route(self, candidate_id: int, name: str | None = None) -> dict | None:
        """Keep a copy of a candidate and its request; None if the candidate is gone. Named by its title by default."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO routes (name, spec, body) "
                "SELECT COALESCE(?, json_extract(c.body, '$.title')), r.spec, c.body "
                "FROM candidates c JOIN requests r ON r.id = c.request_id WHERE c.id = ?",
                (name, candidate_id),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(f"SELECT {_ROUTE_COLUMNS} FROM routes WHERE id = ?", (cursor.lastrowid,))
            return _route(row.fetchone())

    def routes(self) -> list[dict]:
        """Saved routes without their bodies, newest first."""
        with self._connect() as connection:
            return [_route(row) for row in connection.execute(f"SELECT {_ROUTE_COLUMNS} FROM routes ORDER BY id DESC")]

    def route(self, route_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(f"SELECT {_ROUTE_COLUMNS}, spec, body FROM routes WHERE id = ?", (route_id,))
            row = row.fetchone()
        return _route(row) | {"spec": row[5], "body": row[6]} if row else None

    def delete_route(self, route_id: int) -> bool:
        with self._connect() as connection:
            return connection.execute("DELETE FROM routes WHERE id = ?", (route_id,)).rowcount > 0


def _read_place(connection: sqlite3.Connection, place_id: int) -> dict | None:
    row = connection.execute(f"SELECT {_PLACE_COLUMNS} FROM places WHERE id = ?", (place_id,)).fetchone()
    return _place(row) if row else None


def _make_default(connection: sqlite3.Connection, place_id: int) -> None:
    connection.execute("UPDATE places SET is_default = (id = ?)", (place_id,))


def _ensure_default(connection: sqlite3.Connection) -> None:
    """Keep exactly one default while any place exists: the oldest takes over a vacancy."""
    if connection.execute("SELECT 1 FROM places WHERE is_default").fetchone() is None:
        connection.execute("UPDATE places SET is_default = 1 WHERE id = (SELECT MIN(id) FROM places)")
