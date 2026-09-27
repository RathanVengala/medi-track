"""
Durable JSON persistence shared by the order store and the escalation queue.

Three guarantees the plain `json.dump` the stores used to do could not give:

1. **Cross-process exclusion.** `threading.Lock` only serialises threads inside
   one interpreter. Run the app under `uvicorn --workers 4` and four processes
   each hold their own lock, read the same file, and the last writer wins - so
   two callers can be sold the same last pack. `file_lock()` takes an OS-level
   advisory lock that every process on the machine respects.

2. **Atomic writes.** Content goes to a temp file in the same directory, is
   flushed to the platter, and is then moved into place with `os.replace`,
   which is atomic. A crash mid-write leaves either the old file or the new
   one, never half of either.

3. **Recovery.** A truncated or hand-edited file used to crash the process at
   import. `load_json()` falls back to the `.bak` written before the previous
   save, and if that is unusable too it quarantines the bad file and starts
   empty rather than taking the service down.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from app.config import settings

logger = logging.getLogger("meditrack.storage")

# Platform-specific advisory locking. Both release automatically when the
# holding process exits, so a killed worker cannot leave a lock behind.
try:  # POSIX
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # Windows
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

try:  # Windows
    import msvcrt

    _HAVE_MSVCRT = True
except ImportError:  # POSIX
    msvcrt = None  # type: ignore[assignment]
    _HAVE_MSVCRT = False


class StorageError(Exception):
    pass


class LockTimeout(StorageError):
    """The write lock was held by another process for too long."""


def _try_lock(handle) -> bool:
    """One non-blocking attempt at an exclusive lock. False if already held."""
    if _HAVE_FCNTL:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise
    if _HAVE_MSVCRT:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            # Windows reports both "already locked" and "lock violation" as
            # OSError with no stable errno, so any failure means "held".
            return False
    return True  # no locking primitive available; single-process only


def _unlock(handle) -> None:
    try:
        if _HAVE_FCNTL:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        elif _HAVE_MSVCRT:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError as exc:  # pragma: no cover - unlocking rarely fails
        logger.warning("Could not release storage lock: %s", exc)


@contextmanager
def file_lock(path: Path, timeout: float | None = None) -> Iterator[None]:
    """
    Hold an exclusive cross-process lock for `path` while the block runs.

    The lock lives on a sibling `.lock` file rather than the data file itself,
    so locking never conflicts with the `os.replace` that swaps the data file.
    """
    timeout = settings.STORE_LOCK_TIMEOUT_SECONDS if timeout is None else timeout
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    handle = open(lock_path, "a+b")
    try:
        deadline = time.monotonic() + timeout
        delay = 0.01
        while True:
            if _try_lock(handle):
                break
            if time.monotonic() >= deadline:
                raise LockTimeout(
                    f"Timed out after {timeout:g}s waiting for the write lock on {path.name}. "
                    "Another worker may be stuck mid-write."
                )
            time.sleep(delay)
            delay = min(delay * 2, 0.2)  # back off rather than spin
        try:
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()


def atomic_write_json(path: Path, payload: dict) -> None:
    """
    Replace `path` with `payload`, keeping the previous contents as `.bak`.

    Must be called while holding `file_lock(path)`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Snapshot the good file before overwriting it; this is what load_json()
    # recovers from if the machine dies during the replace below.
    if path.exists():
        try:
            backup = path.with_name(path.name + ".bak")
            backup.write_bytes(path.read_bytes())
        except OSError as exc:  # a missing backup must not block the write
            logger.warning("Could not refresh backup for %s: %s", path.name, exc)

    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())  # the rename is only atomic if the data is durable
    os.replace(tmp, path)


def _quarantine(path: Path) -> Path | None:
    """Move an unreadable file aside so the next start is clean."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        os.replace(path, target)
        return target
    except OSError as exc:  # pragma: no cover - only on a locked/readonly file
        logger.error("Could not quarantine %s: %s", path, exc)
        return None


def load_json(path: Path, default: dict | None = None) -> dict:
    """
    Read a JSON store, recovering from a corrupt file instead of crashing.

    Order of preference: the file itself, then its `.bak`, then `default`. A
    file that cannot be parsed is renamed to `<name>.corrupt-<timestamp>` and
    the failure is logged at ERROR so it is visible in the console rather than
    silently swallowed.
    """
    default = {} if default is None else default
    if not path.exists():
        return dict(default)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
        return data
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError, OSError) as exc:
        logger.error("%s is unreadable (%s); attempting recovery.", path.name, exc)

    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        try:
            with backup.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                quarantined = _quarantine(path)
                logger.warning(
                    "Recovered %s from %s. The damaged file was kept as %s.",
                    path.name,
                    backup.name,
                    quarantined.name if quarantined else "(not kept)",
                )
                return data
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError, OSError) as exc:
            logger.error("Backup %s is unreadable too (%s).", backup.name, exc)

    quarantined = _quarantine(path)
    logger.error(
        "Starting with an empty %s. The damaged file was kept as %s for inspection.",
        path.name,
        quarantined.name if quarantined else "(not kept)",
    )
    return dict(default)
