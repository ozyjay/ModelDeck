"""Shared SQLite connection and failure policy for local operational state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

BUSY_TIMEOUT_MS = 2000


def install_persistence_handler(app) -> None:
    """Expose operational storage failures without leaking SQL or file paths."""
    from fastapi.responses import JSONResponse

    @app.middleware("http")
    async def startup_failure(request, call_next):
        error = getattr(app.state, "persistence_startup_error", None)
        if error is not None:
            return JSONResponse(
                {"status": "degraded", "error": error.as_dict(), "cloud_fallback_attempted": False},
                status_code=503,
            )
        return await call_next(request)

    @app.exception_handler(PersistenceError)
    async def persistence_unavailable(request, error):
        return JSONResponse(
            {"status": "degraded", "error": error.as_dict(), "cloud_fallback_attempted": False},
            status_code=503,
        )


class PersistenceError(RuntimeError):
    """A storage failure, distinct from absent records or invalid user input."""

    def __init__(self, code: str, operation: str = "database") -> None:
        self.code = code
        self.operation = operation
        super().__init__(f"Local persistence is unavailable ({code})")

    def as_dict(self) -> dict:
        return {"component": "persistence", "code": self.code, "operation": self.operation}


class PersistenceConstraintError(PersistenceError, sqlite3.IntegrityError):
    """Storage constraint failure; still catchable by existing conflict handlers."""


@contextmanager
def connect_database(
    path: Path, *, create: bool = False, readonly: bool = False
) -> Iterator[sqlite3.Connection]:
    """Open a bounded transaction and always close its connection.

    Ordinary reads and writes require an existing database. Only explicit
    initialisation may create one; a missing file is never an empty registry.
    Integrity errors remain available to callers implementing conflict handling.
    """
    connection = None
    try:
        mode = "ro" if readonly else "rwc" if create else "rw"
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode={mode}", uri=True, timeout=BUSY_TIMEOUT_MS / 1000
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        with connection:
            yield connection
    except sqlite3.IntegrityError as error:
        raise PersistenceConstraintError("database_constraint") from error
    except sqlite3.Error as error:
        category = getattr(error, "sqlite_errorcode", 0) & 0xFF
        code = {
            sqlite3.SQLITE_BUSY: "database_busy",
            sqlite3.SQLITE_LOCKED: "database_locked",
            sqlite3.SQLITE_CORRUPT: "database_corrupt",
            sqlite3.SQLITE_NOTADB: "database_corrupt",
            sqlite3.SQLITE_CANTOPEN: "database_unavailable",
        }.get(category, "database_error")
        raise PersistenceError(code) from error
    finally:
        if connection is not None:
            connection.close()
