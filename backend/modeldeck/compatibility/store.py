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


def evidence_fingerprint(evidence: Mapping[str, Any]) -> str:
    canonical = {field: evidence.get(field) for field in FINGERPRINT_FIELDS}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class CompatibilityStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialise(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as database:
            database.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS model_profiles (
                    id TEXT PRIMARY KEY, document_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_cache_policy (
                    model_id TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    allowed INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model_id, revision)
                );
                CREATE TABLE IF NOT EXISTS gateway_provider_selection (
                    alias TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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
                CREATE TABLE IF NOT EXISTS worker_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    worker_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS presets (
                    id TEXT PRIMARY KEY, document_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                """
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

    def list_model_profiles(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with sqlite3.connect(self.path) as database:
                rows = database.execute("SELECT document_json FROM model_profiles ORDER BY id").fetchall()
        except sqlite3.OperationalError:
            return []
        profiles = []
        for (document_json,) in rows:
            try:
                document = json.loads(document_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(document, dict):
                profiles.append(document)
        return profiles

    def save_model_profile(self, profile: Mapping[str, Any]) -> None:
        profile_id = str(profile["id"])
        updated_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO model_profiles (id, document_json, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET document_json = excluded.document_json, "
                "updated_at = excluded.updated_at",
                (profile_id, json.dumps(dict(profile), sort_keys=True), updated_at),
            )

    def delete_model_profile(self, profile_id: str) -> bool:
        with sqlite3.connect(self.path) as database:
            cursor = database.execute("DELETE FROM model_profiles WHERE id = ?", (profile_id,))
        return cursor.rowcount > 0

    def list_model_cache_policy(self) -> dict[tuple[str, str], bool]:
        if not self.path.exists():
            return {}
        try:
            with sqlite3.connect(self.path) as database:
                rows = database.execute(
                    "SELECT model_id, revision, allowed FROM model_cache_policy"
                ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {(str(row[0]), str(row[1])): bool(row[2]) for row in rows}

    def model_cache_allowed(self, model_id: str, revision: str) -> bool:
        return self.list_model_cache_policy().get((model_id, revision), True)

    def set_model_cache_allowed(self, model_id: str, revision: str, *, allowed: bool) -> None:
        updated_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO model_cache_policy (model_id, revision, allowed, updated_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(model_id, revision) DO UPDATE SET "
                "allowed = excluded.allowed, updated_at = excluded.updated_at",
                (model_id, revision, int(allowed), updated_at),
            )

    def gateway_provider_selection(self, alias: str) -> str | None:
        if not self.path.exists():
            return None
        try:
            with sqlite3.connect(self.path) as database:
                row = database.execute(
                    "SELECT profile_id FROM gateway_provider_selection WHERE alias = ?",
                    (alias,),
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        return str(row[0]) if row is not None else None

    def set_gateway_provider_selection(self, alias: str, profile_id: str) -> None:
        updated_at = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.path) as database:
            database.execute(
                "INSERT INTO gateway_provider_selection (alias, profile_id, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(alias) DO UPDATE SET "
                "profile_id = excluded.profile_id, updated_at = excluded.updated_at",
                (alias, profile_id, updated_at),
            )

    def record_test(
        self,
        evidence: Mapping[str, Any],
        *,
        result: str,
        failure_class: str | None = None,
    ) -> dict[str, Any]:
        tested_at = datetime.now(UTC).isoformat()
        fingerprint = evidence_fingerprint(evidence)
        document = dict(evidence)
        document.update(
            {
                "result": result,
                "failure_class": failure_class,
                "tested_at": tested_at,
            }
        )
        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                "INSERT INTO compatibility_tests "
                "(fingerprint, result, failure_class, evidence_json, tested_at) VALUES (?, ?, ?, ?, ?)",
                (
                    fingerprint,
                    result,
                    failure_class,
                    json.dumps(document, sort_keys=True, default=str),
                    tested_at,
                ),
            )
            test_id = int(cursor.lastrowid)
        return {
            "id": test_id,
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
            evidence = json.loads(row[3])
            evidence.update(dict(updates))
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
