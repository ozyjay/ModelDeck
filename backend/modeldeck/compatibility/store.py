from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FINGERPRINT_FIELDS = (
    "hardware_profile",
    "fedora_version",
    "kernel",
    "gpu",
    "gpu_architecture",
    "rocm_version",
    "torch_version",
    "transformers_version",
    "vllm_version",
    "model_id",
    "model_revision",
    "quantisation",
    "dtype",
    "runtime",
    "environment_overrides",
)


class LegacyDatabaseError(RuntimeError):
    pass


def evidence_fingerprint(evidence: Mapping[str, Any]) -> str:
    canonical = {field: evidence.get(field) for field in FINGERPRINT_FIELDS}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class CompatibilityStore:
    """SQLite persistence for the v2 ModelDeck domain and compatibility evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialise(self) -> None:
        self.initialise_v2()

    def initialise_v2(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as database:
            tables = {
                str(row[0])
                for row in database.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            }
            if tables and "schema_metadata" not in tables:
                raise LegacyDatabaseError(
                    "This is a legacy ModelDeck database. Run scripts/cutover_v2.ps1 before starting."
                )
            if "schema_metadata" in tables:
                row = database.execute(
                    "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
                ).fetchone()
                if row is None or str(row[0]) != "2":
                    raise LegacyDatabaseError("The ModelDeck database schema is not version 2")
            database.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS configuration_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE,
                    document_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived_at TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    draft_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS event_revisions (
                    event_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    document_json TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    PRIMARY KEY (event_id, revision),
                    FOREIGN KEY (event_id) REFERENCES events(id)
                );
                CREATE TABLE IF NOT EXISTS active_event (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    event_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    routing_json TEXT NOT NULL,
                    published_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_cache_policy (
                    model_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    allowed INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model_id, revision)
                );
                CREATE TABLE IF NOT EXISTS compatibility_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL,
                    result TEXT NOT NULL,
                    failure_class TEXT,
                    evidence_json TEXT NOT NULL,
                    tested_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS compatibility_fingerprint_idx
                    ON compatibility_tests(fingerprint, tested_at);
                CREATE UNIQUE INDEX IF NOT EXISTS workers_active_name_idx
                    ON workers(name COLLATE NOCASE) WHERE archived_at IS NULL;
                """
            )
            now = _now()
            database.execute(
                "INSERT INTO schema_metadata (key, value, updated_at) VALUES ('schema_version', '2', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (now,),
            )

    def list_workers(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        query = "SELECT document_json, created_at, updated_at, archived_at FROM workers"
        if not include_archived:
            query += " WHERE archived_at IS NULL"
        query += " ORDER BY name COLLATE NOCASE, id"
        try:
            with sqlite3.connect(self.path) as database:
                rows = database.execute(query).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {
                "definition": json.loads(row[0]),
                "created_at": row[1],
                "updated_at": row[2],
                "archived_at": row[3],
            }
            for row in rows
        ]

    def get_worker_definition(self, worker_id: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with sqlite3.connect(self.path) as database:
            row = database.execute(
                "SELECT document_json, created_at, updated_at, archived_at FROM workers WHERE id = ?",
                (worker_id,),
            ).fetchone()
        return (
            {
                "definition": json.loads(row[0]),
                "created_at": row[1],
                "updated_at": row[2],
                "archived_at": row[3],
            }
            if row
            else None
        )

    def save_worker_definition(self, document: Mapping[str, Any]) -> dict[str, Any]:
        now = _now()
        worker_id = str(document["id"])
        try:
            with sqlite3.connect(self.path) as database:
                database.execute(
                    "INSERT INTO workers (id, name, document_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                    "name = excluded.name, document_json = excluded.document_json, "
                    "updated_at = excluded.updated_at",
                    (
                        worker_id,
                        str(document["name"]),
                        json.dumps(dict(document), sort_keys=True),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("A Worker with that name already exists") from error
        return self.get_worker_definition(worker_id)  # type: ignore[return-value]

    def archive_worker(self, worker_id: str) -> bool:
        now = _now()
        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                "UPDATE workers SET archived_at = ?, updated_at = ? WHERE id = ? AND archived_at IS NULL",
                (now, now, worker_id),
            )
        return cursor.rowcount > 0

    def delete_worker_definition(self, worker_id: str) -> bool:
        with sqlite3.connect(self.path) as database:
            cursor = database.execute("DELETE FROM workers WHERE id = ?", (worker_id,))
        return cursor.rowcount > 0

    def list_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with sqlite3.connect(self.path) as database:
            rows = database.execute(
                "SELECT events.id, events.draft_json, events.created_at, events.updated_at, "
                "active.event_id IS NOT NULL, active.revision, "
                "(SELECT MAX(revision) FROM event_revisions WHERE event_id = events.id) "
                "FROM events LEFT JOIN active_event AS active "
                "ON active.singleton_id = 1 AND active.event_id = events.id "
                "ORDER BY json_extract(events.draft_json, '$.name') COLLATE NOCASE"
            ).fetchall()
        return [
            {
                "definition": json.loads(row[1]),
                "created_at": row[2],
                "updated_at": row[3],
                "active": bool(row[4]),
                "active_revision": int(row[5]) if row[5] is not None else None,
                "latest_revision": int(row[6]) if row[6] is not None else None,
            }
            for row in rows
        ]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        return next((item for item in self.list_events() if item["definition"]["id"] == event_id), None)

    def save_event_draft(self, document: Mapping[str, Any]) -> dict[str, Any]:
        now = _now()
        event_id = str(document["id"])
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO events (id, draft_json, created_at, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET draft_json = excluded.draft_json, "
                "updated_at = excluded.updated_at",
                (event_id, json.dumps(dict(document), sort_keys=True), now, now),
            )
        return self.get_event(event_id)  # type: ignore[return-value]

    def delete_event(self, event_id: str) -> bool:
        with sqlite3.connect(self.path) as database:
            active = database.execute(
                "SELECT 1 FROM active_event WHERE singleton_id = 1 AND event_id = ?", (event_id,)
            ).fetchone()
            revision = database.execute(
                "SELECT 1 FROM event_revisions WHERE event_id = ? LIMIT 1", (event_id,)
            ).fetchone()
            if active or revision:
                raise RuntimeError("Published Events cannot be deleted")
            cursor = database.execute("DELETE FROM events WHERE id = ?", (event_id,))
        return cursor.rowcount > 0

    def list_event_revisions(self, event_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as database:
            rows = database.execute(
                "SELECT revision, document_json, published_at FROM event_revisions "
                "WHERE event_id = ? ORDER BY revision DESC",
                (event_id,),
            ).fetchall()
            active = database.execute(
                "SELECT revision FROM active_event WHERE singleton_id = 1 AND event_id = ?",
                (event_id,),
            ).fetchone()
        active_revision = int(active[0]) if active else None
        return [
            {
                "definition": json.loads(row[1]),
                "revision": int(row[0]),
                "published_at": row[2],
                "active": active_revision == int(row[0]),
            }
            for row in rows
        ]

    def get_event_revision(self, event_id: str, revision: int) -> dict[str, Any] | None:
        return next(
            (item for item in self.list_event_revisions(event_id) if item["revision"] == revision),
            None,
        )

    def publish_event(self, document: Mapping[str, Any], routing: Mapping[str, Any]) -> dict[str, Any]:
        event_id = str(document["id"])
        published_at = _now()
        with sqlite3.connect(self.path) as database:
            row = database.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM event_revisions WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            revision = int(row[0]) + 1
            database.execute(
                "INSERT INTO event_revisions (event_id, revision, document_json, published_at) "
                "VALUES (?, ?, ?, ?)",
                (event_id, revision, json.dumps(dict(document), sort_keys=True), published_at),
            )
            self._set_active(database, event_id, revision, routing, published_at)
        return self.get_event_revision(event_id, revision)  # type: ignore[return-value]

    def activate_event_revision(
        self, event_id: str, revision: int, routing: Mapping[str, Any]
    ) -> dict[str, Any]:
        record = self.get_event_revision(event_id, revision)
        if record is None:
            raise KeyError("Unknown Event revision")
        with sqlite3.connect(self.path) as database:
            self._set_active(database, event_id, revision, routing, _now())
        return record

    @staticmethod
    def _set_active(
        database: sqlite3.Connection,
        event_id: str,
        revision: int,
        routing: Mapping[str, Any],
        published_at: str,
    ) -> None:
        snapshot = {**dict(routing), "revision": revision}
        database.execute(
            "INSERT INTO active_event "
            "(singleton_id, event_id, revision, routing_json, published_at) "
            "VALUES (1, ?, ?, ?, ?) ON CONFLICT(singleton_id) DO UPDATE SET "
            "event_id = excluded.event_id, revision = excluded.revision, "
            "routing_json = excluded.routing_json, published_at = excluded.published_at",
            (event_id, revision, json.dumps(snapshot, sort_keys=True), published_at),
        )

    def active_routing_snapshot(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            with sqlite3.connect(self.path) as database:
                row = database.execute(
                    "SELECT routing_json FROM active_event WHERE singleton_id = 1"
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        return json.loads(row[0]) if row else None

    def discard_event_draft(self, event_id: str) -> dict[str, Any]:
        revisions = self.list_event_revisions(event_id)
        if not revisions:
            raise RuntimeError("An unpublished Event has no published revision to restore")
        return self.save_event_draft(revisions[0]["definition"])

    def rebind_event_drafts(self, old_worker_id: str, new_worker_id: str) -> list[str]:
        changed: list[str] = []
        with sqlite3.connect(self.path) as database:
            rows = database.execute("SELECT id, draft_json FROM events").fetchall()
            for event_id, document_json in rows:
                document = json.loads(document_json)
                touched = False
                for route in document.get("routes", []):
                    if old_worker_id in route.get("worker_ids", []):
                        route["worker_ids"] = [
                            new_worker_id if item == old_worker_id else item for item in route["worker_ids"]
                        ]
                        touched = True
                if touched:
                    database.execute(
                        "UPDATE events SET draft_json = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(document, sort_keys=True), _now(), event_id),
                    )
                    changed.append(str(event_id))
        return changed

    def list_model_cache_policy(self) -> dict[tuple[str, str], bool]:
        if not self.path.exists():
            return {}
        with sqlite3.connect(self.path) as database:
            rows = database.execute("SELECT model_id, revision, allowed FROM model_cache_policy").fetchall()
        return {(str(row[0]), str(row[1])): bool(row[2]) for row in rows}

    def model_cache_allowed(self, model_id: str, revision: str) -> bool:
        return self.list_model_cache_policy().get((model_id, revision), True)

    def set_model_cache_allowed(self, model_id: str, revision: str, *, allowed: bool) -> None:
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO model_cache_policy (model_id, revision, allowed, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(model_id, revision) DO UPDATE SET "
                "allowed = excluded.allowed, updated_at = excluded.updated_at",
                (model_id, revision, int(allowed), _now()),
            )

    def list_tests(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with sqlite3.connect(self.path) as database:
            rows = database.execute(
                "SELECT id, fingerprint, result, failure_class, evidence_json, tested_at "
                "FROM compatibility_tests ORDER BY id DESC"
            ).fetchall()
        return [
            {
                "id": row[0],
                "fingerprint": row[1],
                "result": row[2],
                "failure_class": row[3],
                "evidence": json.loads(row[4]),
                "tested_at": row[5],
            }
            for row in rows
        ]

    def record_test(
        self, evidence: Mapping[str, Any], *, result: str, failure_class: str | None = None
    ) -> dict[str, Any]:
        tested_at = _now()
        fingerprint = evidence_fingerprint(evidence)
        document = {
            **dict(evidence),
            "result": result,
            "failure_class": failure_class,
            "tested_at": tested_at,
        }
        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                "INSERT INTO compatibility_tests "
                "(fingerprint, result, failure_class, evidence_json, tested_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    fingerprint,
                    result,
                    failure_class,
                    json.dumps(document, sort_keys=True, default=str),
                    tested_at,
                ),
            )
        return {
            "id": int(cursor.lastrowid),
            "fingerprint": fingerprint,
            "result": result,
            "failure_class": failure_class,
            "evidence": document,
            "tested_at": tested_at,
        }

    def update_test_evidence(self, test_id: int, updates: Mapping[str, Any]) -> dict[str, Any]:
        with sqlite3.connect(self.path) as database:
            row = database.execute(
                "SELECT fingerprint, result, failure_class, evidence_json, tested_at "
                "FROM compatibility_tests WHERE id = ?",
                (test_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown compatibility test: {test_id}")
            evidence = {**json.loads(row[3]), **dict(updates)}
            database.execute(
                "UPDATE compatibility_tests SET evidence_json = ? WHERE id = ?",
                (json.dumps(evidence, sort_keys=True, default=str), test_id),
            )
        return {
            "id": test_id,
            "fingerprint": row[0],
            "result": row[1],
            "failure_class": row[2],
            "evidence": evidence,
            "tested_at": row[4],
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()
