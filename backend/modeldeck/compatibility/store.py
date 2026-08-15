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
    "capability_id",
    "protocol_contract_id",
    "runtime_template_id",
    "runtime_template_version",
    "worker_configuration_fingerprint",
    "environment_overrides",
)


class LegacyDatabaseError(RuntimeError):
    pass


def evidence_fingerprint(evidence: Mapping[str, Any]) -> str:
    canonical = {field: evidence.get(field) for field in FINGERPRINT_FIELDS}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _ensure_no_active_capability_collisions(
    database: sqlite3.Connection, profile_id: str, snapshot: Mapping[str, Any]
) -> None:
    """Reject ambiguous public model IDs before an activation becomes visible."""

    requested = {
        str(capability.get("public_name", "")).casefold()
        for capability in snapshot.get("capabilities", [])
        if str(capability.get("public_name", ""))
    }
    if not requested:
        return
    rows = database.execute(
        "SELECT profile_id, routing_json FROM active_routing_profiles WHERE profile_id != ?",
        (profile_id,),
    ).fetchall()
    conflicts: list[str] = []
    for _active_profile_id, routing_json in rows:
        active_names = {
            str(capability.get("public_name", "")).casefold()
            for capability in json.loads(routing_json).get("capabilities", [])
        }
        conflicts.extend(sorted(requested & active_names))
    if conflicts:
        raise ValueError(
            "Active Routing Profiles must use unique API Model IDs; conflicts: "
            + ", ".join(sorted(set(conflicts)))
        )


class CompatibilityStore:
    """SQLite persistence for routing profiles and compatibility evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialise(self) -> None:
        self.initialise_v4()

    def initialise_v3(self) -> None:
        """Compatibility alias for callers creating a new current database."""
        self.initialise_v4()

    def initialise_v4(self) -> None:
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
                if row is None:
                    raise LegacyDatabaseError("The ModelDeck database schema has no version")
                if str(row[0]) == "2":
                    raise LegacyDatabaseError("Run scripts/migrate_v2_to_v3.ps1 before starting ModelDeck.")
                if str(row[0]) == "3":
                    raise LegacyDatabaseError("Run scripts/migrate_v3_to_v4.ps1 before starting ModelDeck.")
                if str(row[0]) != "4":
                    raise LegacyDatabaseError("The ModelDeck database schema is not version 4")
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
                CREATE TABLE IF NOT EXISTS routing_profiles (
                    id TEXT PRIMARY KEY,
                    draft_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS routing_profile_revisions (
                    profile_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    document_json TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    PRIMARY KEY (profile_id, revision),
                    FOREIGN KEY (profile_id) REFERENCES routing_profiles(id)
                );
                CREATE TABLE IF NOT EXISTS active_routing_profile (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    profile_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    routing_json TEXT NOT NULL,
                    published_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_routing_profiles (
                    profile_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    routing_json TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES routing_profiles(id)
                );
                CREATE TABLE IF NOT EXISTS gateway_job_assignments (
                    job_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    capability_name TEXT NOT NULL,
                    protocol_contract TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS route_tool_calling_rehearsals (
                    profile_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    capability_id TEXT NOT NULL,
                    supported INTEGER NOT NULL,
                    rehearsed INTEGER NOT NULL,
                    last_rehearsal TEXT,
                    failure_code TEXT,
                    evidence_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (profile_id, revision, capability_id)
                );
                CREATE TABLE IF NOT EXISTS model_cache_policy (
                    model_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    allowed INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model_id, revision)
                );
                CREATE TABLE IF NOT EXISTS model_capability_policy (
                    model_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    allowed INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model_id, revision, capability_id)
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
                "INSERT INTO schema_metadata (key, value, updated_at) VALUES ('schema_version', '4', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (now,),
            )
            # Keep installations migrated from the original singleton activation
            # model live with an explicit set of active profiles.
            # The old table remains readable for local downgrade diagnostics.
            database.execute(
                "INSERT OR IGNORE INTO active_routing_profiles "
                "(profile_id, revision, routing_json, published_at) "
                "SELECT profile_id, revision, routing_json, published_at "
                "FROM active_routing_profile WHERE singleton_id = 1"
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

    def list_routing_profiles(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with sqlite3.connect(self.path) as database:
            rows = database.execute(
                "SELECT profiles.id, profiles.draft_json, profiles.created_at, profiles.updated_at, "
                "active.profile_id IS NOT NULL, active.revision, "
                "(SELECT MAX(revision) FROM routing_profile_revisions WHERE profile_id = profiles.id) "
                "FROM routing_profiles AS profiles LEFT JOIN active_routing_profiles AS active "
                "ON active.profile_id = profiles.id "
                "ORDER BY json_extract(profiles.draft_json, '$.name') COLLATE NOCASE"
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

    def get_routing_profile(self, profile_id: str) -> dict[str, Any] | None:
        return next(
            (item for item in self.list_routing_profiles() if item["definition"]["id"] == profile_id),
            None,
        )

    def save_routing_profile_draft(self, document: Mapping[str, Any]) -> dict[str, Any]:
        now = _now()
        profile_id = str(document["id"])
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO routing_profiles (id, draft_json, created_at, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET draft_json = excluded.draft_json, "
                "updated_at = excluded.updated_at",
                (profile_id, json.dumps(dict(document), sort_keys=True), now, now),
            )
        return self.get_routing_profile(profile_id)  # type: ignore[return-value]

    def delete_routing_profile(self, profile_id: str) -> bool:
        with sqlite3.connect(self.path) as database:
            active = database.execute(
                "SELECT 1 FROM active_routing_profiles WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            revision = database.execute(
                "SELECT 1 FROM routing_profile_revisions WHERE profile_id = ? LIMIT 1", (profile_id,)
            ).fetchone()
            if active or revision:
                raise RuntimeError("Published Routing Profiles cannot be deleted")
            cursor = database.execute("DELETE FROM routing_profiles WHERE id = ?", (profile_id,))
        return cursor.rowcount > 0

    def list_routing_profile_revisions(self, profile_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as database:
            rows = database.execute(
                "SELECT revision, document_json, published_at FROM routing_profile_revisions "
                "WHERE profile_id = ? ORDER BY revision DESC",
                (profile_id,),
            ).fetchall()
            active = database.execute(
                "SELECT revision FROM active_routing_profiles WHERE profile_id = ?",
                (profile_id,),
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

    def get_routing_profile_revision(self, profile_id: str, revision: int) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.list_routing_profile_revisions(profile_id)
                if item["revision"] == revision
            ),
            None,
        )

    def publish_routing_profile(
        self, document: Mapping[str, Any], routing: Mapping[str, Any]
    ) -> dict[str, Any]:
        profile_id = str(document["id"])
        published_at = _now()
        with sqlite3.connect(self.path) as database:
            row = database.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM routing_profile_revisions WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            revision = int(row[0]) + 1
            database.execute(
                "INSERT INTO routing_profile_revisions (profile_id, revision, document_json, published_at) "
                "VALUES (?, ?, ?, ?)",
                (profile_id, revision, json.dumps(dict(document), sort_keys=True), published_at),
            )
            self._set_active_routing_profile(database, profile_id, revision, routing, published_at)
        return self.get_routing_profile_revision(profile_id, revision)  # type: ignore[return-value]

    def activate_routing_profile_revision(
        self, profile_id: str, revision: int, routing: Mapping[str, Any]
    ) -> dict[str, Any]:
        record = self.get_routing_profile_revision(profile_id, revision)
        if record is None:
            raise KeyError("Unknown Routing Profile revision")
        with sqlite3.connect(self.path) as database:
            self._set_active_routing_profile(database, profile_id, revision, routing, _now())
        return record

    @staticmethod
    def _set_active_routing_profile(
        database: sqlite3.Connection,
        profile_id: str,
        revision: int,
        routing: Mapping[str, Any],
        published_at: str,
    ) -> None:
        snapshot = {**dict(routing), "revision": revision}
        _ensure_no_active_capability_collisions(database, profile_id, snapshot)
        database.execute(
            "INSERT INTO active_routing_profiles "
            "(profile_id, revision, routing_json, published_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(profile_id) DO UPDATE SET "
            "revision = excluded.revision, routing_json = excluded.routing_json, "
            "published_at = excluded.published_at",
            (profile_id, revision, json.dumps(snapshot, sort_keys=True), published_at),
        )
        # Retain the latest activation in the legacy singleton table. New code never
        # reads it for routing, but it makes a rollback to an older build diagnosable.
        database.execute(
            "INSERT INTO active_routing_profile "
            "(singleton_id, profile_id, revision, routing_json, published_at) "
            "VALUES (1, ?, ?, ?, ?) ON CONFLICT(singleton_id) DO UPDATE SET "
            "profile_id = excluded.profile_id, revision = excluded.revision, "
            "routing_json = excluded.routing_json, published_at = excluded.published_at",
            (profile_id, revision, json.dumps(snapshot, sort_keys=True), published_at),
        )

    def active_routing_snapshot(self) -> dict[str, Any] | None:
        """Compatibility accessor for callers that require exactly one active profile."""
        snapshots = self.active_routing_snapshots()
        return snapshots[0] if len(snapshots) == 1 else None

    def active_routing_snapshots(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with sqlite3.connect(self.path) as database:
                rows = database.execute(
                    "SELECT routing_json FROM active_routing_profiles ORDER BY published_at, profile_id"
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [json.loads(row[0]) for row in rows]

    def deactivate_routing_profile(self, profile_id: str) -> bool:
        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                "DELETE FROM active_routing_profiles WHERE profile_id = ?", (profile_id,)
            )
        return cursor.rowcount > 0

    def route_tool_calling_state(
        self, profile_id: str | None, revision: int | None, capability_id: str | None
    ) -> dict[str, Any]:
        """Return only coarse, revision-scoped tool-calling rehearsal state."""

        default = {
            "supported": False,
            "rehearsed": False,
            "last_rehearsal": None,
            "failure_code": None,
        }
        if not profile_id or revision is None or not capability_id or not self.path.exists():
            return default
        try:
            with sqlite3.connect(self.path) as database:
                row = database.execute(
                    "SELECT supported, rehearsed, last_rehearsal, failure_code "
                    "FROM route_tool_calling_rehearsals "
                    "WHERE profile_id = ? AND revision = ? AND capability_id = ?",
                    (profile_id, revision, capability_id),
                ).fetchone()
        except sqlite3.OperationalError:
            return default
        return (
            {
                "supported": bool(row[0]),
                "rehearsed": bool(row[1]),
                "last_rehearsal": row[2],
                "failure_code": row[3],
            }
            if row
            else default
        )

    def save_route_tool_calling_rehearsal(
        self,
        profile_id: str,
        revision: int,
        capability_id: str,
        *,
        supported: bool,
        failure_code: str | None,
        evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist bounded probe facts without retaining prompts or model output."""

        rehearsed_at = _now()
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO route_tool_calling_rehearsals "
                "(profile_id, revision, capability_id, supported, rehearsed, last_rehearsal, "
                "failure_code, evidence_json, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?) "
                "ON CONFLICT(profile_id, revision, capability_id) DO UPDATE SET "
                "supported = excluded.supported, rehearsed = excluded.rehearsed, "
                "last_rehearsal = excluded.last_rehearsal, failure_code = excluded.failure_code, "
                "evidence_json = excluded.evidence_json, updated_at = excluded.updated_at",
                (
                    profile_id,
                    revision,
                    capability_id,
                    int(supported),
                    rehearsed_at,
                    failure_code,
                    json.dumps(dict(evidence), sort_keys=True),
                    rehearsed_at,
                ),
            )
        return self.route_tool_calling_state(profile_id, revision, capability_id)

    def discard_routing_profile_draft(self, profile_id: str) -> dict[str, Any]:
        revisions = self.list_routing_profile_revisions(profile_id)
        if not revisions:
            raise RuntimeError("An unpublished Routing Profile has no published revision to restore")
        return self.save_routing_profile_draft(revisions[0]["definition"])

    def rebind_routing_profile_drafts(self, old_worker_id: str, new_worker_id: str) -> list[str]:
        changed: list[str] = []
        with sqlite3.connect(self.path) as database:
            rows = database.execute("SELECT id, draft_json FROM routing_profiles").fetchall()
            for profile_id, document_json in rows:
                document = json.loads(document_json)
                touched = False
                for capability in document.get("capabilities", []):
                    if old_worker_id in capability.get("worker_ids", []):
                        capability["worker_ids"] = [
                            new_worker_id if item == old_worker_id else item
                            for item in capability["worker_ids"]
                        ]
                        touched = True
                if touched:
                    database.execute(
                        "UPDATE routing_profiles SET draft_json = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(document, sort_keys=True), _now(), profile_id),
                    )
                    changed.append(str(profile_id))
        return changed

    def save_gateway_job_assignment(
        self, job_id: str, worker_id: str, capability_name: str, protocol_contract: str
    ) -> None:
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT OR REPLACE INTO gateway_job_assignments "
                "(job_id, worker_id, capability_name, protocol_contract, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, worker_id, capability_name, protocol_contract, _now()),
            )

    def get_gateway_job_assignment(self, job_id: str) -> dict[str, str] | None:
        with sqlite3.connect(self.path) as database:
            row = database.execute(
                "SELECT worker_id, capability_name, protocol_contract "
                "FROM gateway_job_assignments WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return (
            {"worker_id": str(row[0]), "capability_name": str(row[1]), "protocol_contract": str(row[2])}
            if row
            else None
        )

    def delete_gateway_job_assignment(self, job_id: str) -> None:
        with sqlite3.connect(self.path) as database:
            database.execute("DELETE FROM gateway_job_assignments WHERE job_id = ?", (job_id,))

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

    def list_model_capability_policy(self) -> dict[tuple[str, str, str], bool]:
        if not self.path.exists():
            return {}
        with sqlite3.connect(self.path) as database:
            rows = database.execute(
                "SELECT model_id, revision, capability_id, allowed FROM model_capability_policy"
            ).fetchall()
        return {(str(row[0]), str(row[1]), str(row[2])): bool(row[3]) for row in rows}

    def model_capability_allowed(self, model_id: str, revision: str, capability_id: str) -> bool:
        return self.list_model_capability_policy().get((model_id, revision, capability_id), False)

    def set_model_capability_allowed(
        self,
        model_id: str,
        revision: str,
        capability_id: str,
        *,
        allowed: bool,
    ) -> None:
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO model_capability_policy "
                "(model_id, revision, capability_id, allowed, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(model_id, revision, capability_id) "
                "DO UPDATE SET allowed = excluded.allowed, updated_at = excluded.updated_at",
                (model_id, revision, capability_id, int(allowed), _now()),
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
