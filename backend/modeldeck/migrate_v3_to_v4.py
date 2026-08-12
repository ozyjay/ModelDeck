"""Explicit, local-only conversion of ModelDeck's v3 database to capability policy v4."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modeldeck.capabilities import (
    capabilities_for_worker,
    capability_id_for_contract,
    worker_cache_identity,
)


def migrate(database_path: Path) -> None:
    if not database_path.is_file():
        raise RuntimeError(f"ModelDeck database does not exist: {database_path}")
    with sqlite3.connect(database_path) as database:
        version_row = database.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if version_row is not None and str(version_row[0]) == "4":
            policy_table = database.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'model_capability_policy'"
            ).fetchone()
            if policy_table is None:
                raise RuntimeError("The ModelDeck v4 capability policy table is missing")
            return
        if version_row is None or str(version_row[0]) != "3":
            raise RuntimeError("The selected database is not a ModelDeck v3 database")
        database.execute("BEGIN IMMEDIATE")
        try:
            database.execute(
                "CREATE TABLE model_capability_policy ("
                "model_id TEXT NOT NULL, revision TEXT NOT NULL, capability_id TEXT NOT NULL, "
                "allowed INTEGER NOT NULL, updated_at TEXT NOT NULL, "
                "PRIMARY KEY (model_id, revision, capability_id))"
            )
            workers = _workers(database)
            allowed: set[tuple[str, str, str]] = set()
            for worker in workers.values():
                if worker.get("archived") is True:
                    continue
                model_id, revision = worker_cache_identity(worker)
                allowed.update(
                    (model_id, revision, capability_id) for capability_id in capabilities_for_worker(worker)
                )
            for document in _current_profile_documents(database):
                for binding in document.get("capabilities", []):
                    capability_id = capability_id_for_contract(str(binding.get("protocol_contract", "")))
                    if capability_id is None:
                        continue
                    for worker_id in binding.get("worker_ids", []):
                        worker = workers.get(str(worker_id))
                        if worker is None:
                            continue
                        model_id, revision = worker_cache_identity(worker)
                        allowed.add((model_id, revision, capability_id))
            now = datetime.now(UTC).isoformat()
            database.executemany(
                "INSERT INTO model_capability_policy "
                "(model_id, revision, capability_id, allowed, updated_at) VALUES (?, ?, ?, 1, ?)",
                [(*identity, now) for identity in sorted(allowed)],
            )
            database.execute(
                "UPDATE schema_metadata SET value = '4', updated_at = ? WHERE key = 'schema_version'",
                (now,),
            )
            database.commit()
        except Exception:
            database.rollback()
            raise


def _workers(database: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    try:
        rows = database.execute("SELECT id, document_json, archived_at FROM workers").fetchall()
    except sqlite3.OperationalError:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for worker_id, document_json, archived_at in rows:
        document = json.loads(document_json)
        document["archived"] = archived_at is not None
        result[str(worker_id)] = document
    return result


def _current_profile_documents(database: sqlite3.Connection) -> list[dict[str, Any]]:
    documents = [
        json.loads(row[0]) for row in database.execute("SELECT draft_json FROM routing_profiles").fetchall()
    ]
    tables = {
        str(row[0])
        for row in database.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    if "active_routing_profiles" in tables:
        documents.extend(
            json.loads(row[0])
            for row in database.execute(
                "SELECT revisions.document_json FROM active_routing_profiles AS active "
                "JOIN routing_profile_revisions AS revisions "
                "ON revisions.profile_id = active.profile_id AND revisions.revision = active.revision"
            ).fetchall()
        )
    elif "active_routing_profile" in tables:
        documents.extend(
            json.loads(row[0])
            for row in database.execute(
                "SELECT revisions.document_json FROM active_routing_profile AS active "
                "JOIN routing_profile_revisions AS revisions "
                "ON revisions.profile_id = active.profile_id AND revisions.revision = active.revision "
                "WHERE active.singleton_id = 1"
            ).fetchall()
        )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a local ModelDeck v3 database to v4.")
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    migrate(args.database)


if __name__ == "__main__":
    main()
