"""Adding rides from the web page: files go to the inbox, a separate process works through them.

The server never loads the map matcher. It only stores what was uploaded and starts
`rpa sync-tracks`, whose result (a new rides file next to the graph) it then notices.
"""

import os
import re
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

MAX_UPLOAD_BYTES = 4 << 30
LOG_LINES = 20
_NAME = re.compile(r"[^/\\\0]{1,200}\.(zip|((gpx|fit|tcx)(\.gz)?))", re.IGNORECASE)

_BAD_NAME = "Принимаются архив Strava (.zip) и треки .gpx, .fit, .tcx (можно сжатые .gz)."
_TOO_LARGE = "Файл больше 4 ГБ."
_BUSY = "Идёт обработка треков. Дождитесь её окончания."


class Refused(Exception):
    """An upload that cannot be taken; `status` is the HTTP code that says why."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Upload:
    """A file being received; `size` counts the bytes written so far."""

    def __init__(self, file) -> None:
        self._file = file
        self.size = 0

    def write(self, chunk: bytes) -> None:
        self.size += len(chunk)
        if self.size > MAX_UPLOAD_BYTES:
            raise Refused(413, _TOO_LARGE)
        self._file.write(chunk)


class TrackImport:
    def __init__(self, inbox: Path, log_path: Path, command: list[str]) -> None:
        self.inbox = Path(inbox)
        self.log_path = Path(log_path)
        self.command = command
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._started: str | None = None
        self._finished: str | None = None

    @contextmanager
    def receiving(self, name: str) -> Iterator[Upload]:
        """Take a file into the inbox under `name`: complete on a clean exit, gone without a trace otherwise."""
        if not _NAME.fullmatch(name) or name.startswith("."):
            raise Refused(400, _BAD_NAME)
        if self._running():
            raise Refused(409, _BUSY)
        self.inbox.mkdir(parents=True, exist_ok=True)
        # The dot keeps a half-received file out of sight of the import.
        partial = self.inbox / f".{name}.part"
        try:
            with partial.open("wb") as file:
                yield Upload(file)
            partial.replace(self.inbox / name)
        finally:
            partial.unlink(missing_ok=True)

    def start(self) -> bool:
        """Start working through the inbox; False if that is already going on."""
        with self._lock:
            if self._running():
                return False
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w") as log:
                self._process = subprocess.Popen(
                    self.command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=os.environ | {"PYTHONUNBUFFERED": "1"},  # so that the log grows while the work goes on
                )
            self._started, self._finished = _now(), None
            return True

    def status(self) -> dict:
        with self._lock:
            process = self._process
            if process is None:
                state = "idle"
            elif process.poll() is None:
                state = "running"
            else:
                state = "done" if process.returncode == 0 else "failed"
                self._finished = self._finished or _now()
            return {
                "state": state,
                "started": self._started,
                "finished": self._finished,
                "log": self._log_tail() if process is not None else [],
                "inbox": self._waiting(),
            }

    def wait(self) -> None:
        if self._process is not None:
            self._process.wait()

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait()

    def _running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _log_tail(self) -> list[str]:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-LOG_LINES:]

    def _waiting(self) -> list[dict]:
        if not self.inbox.is_dir():
            return []
        files = (path for path in sorted(self.inbox.iterdir()) if path.is_file() and not path.name.startswith("."))
        return [{"name": path.name, "size": path.stat().st_size} for path in files]


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
