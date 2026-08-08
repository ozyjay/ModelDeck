"""Explicit, local-only conversion of ModelDeck's v2 Event database to v3 profiles."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from modeldeck.domain import RoutingProfile, routing_snapshot


def _profile_document(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": event["id"],
        "name": event["name"],
        "description": event.get("description", ""),
        "qualification": event.get("qualification", "compatible"),
        "capabilities": [
            {
                "id": route["id"],
                "display_name": route["display_name"],
                "public_name": route["public_name"],
                "protocol_contract": route["protocol_contract"],
                "worker_ids": list(route["worker_ids"]),
            }
            for route in event.get("routes", [])
        ],
    }


def migrate(database_path: Path) -> None:
    if not database_path.is_file():
        raise RuntimeError(f"ModelDeck database does not exist: {database_path}")
    with sqlite3.connect(database_path) as database:
        version_row = database.execute(
            "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if version_row is None or str(version_row[0]) != "2":
            raise RuntimeError("The selected database is not a ModelDeck v2 database")
        database.execute("BEGIN IMMEDIATE")
        try:
            for statement in (
                "CREATE TABLE routing_profiles (id TEXT PRIMARY KEY, draft_json TEXT NOT NULL, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
                "CREATE TABLE routing_profile_revisions (profile_id TEXT NOT NULL, "
                "revision INTEGER NOT NULL, "
                "document_json TEXT NOT NULL, published_at TEXT NOT NULL, "
                "PRIMARY KEY (profile_id, revision), "
                "FOREIGN KEY (profile_id) REFERENCES routing_profiles(id))",
                "CREATE TABLE active_routing_profile (singleton_id INTEGER PRIMARY KEY "
                "CHECK (singleton_id = 1), profile_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                "routing_json TEXT NOT NULL, published_at TEXT NOT NULL)",
                "CREATE TABLE gateway_job_assignments (job_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL, "
                "capability_name TEXT NOT NULL, protocol_contract TEXT NOT NULL, created_at TEXT NOT NULL)",
            ):
                database.execute(statement)
            drafts = database.execute("SELECT id, draft_json, created_at, updated_at FROM events").fetchall()
            for profile_id, document_json, created_at, updated_at in drafts:
                document = RoutingProfile.model_validate(_profile_document(json.loads(document_json)))
                database.execute(
                    "INSERT INTO routing_profiles (id, draft_json, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (profile_id, document.model_dump_json(), created_at, updated_at),
                )
            revisions = database.execute(
                "SELECT event_id, revision, document_json, published_at FROM event_revisions"
            ).fetchall()
            profiles_by_revision: dict[tuple[str, int], RoutingProfile] = {}
            for profile_id, revision, document_json, published_at in revisions:
                document = RoutingProfile.model_validate(_profile_document(json.loads(document_json)))
                profiles_by_revision[(str(profile_id), int(revision))] = document
                database.execute(
                    "INSERT INTO routing_profile_revisions "
                    "(profile_id, revision, document_json, published_at) "
                    "VALUES (?, ?, ?, ?)",
                    (profile_id, revision, document.model_dump_json(), published_at),
                )
            active = database.execute(
                "SELECT event_id, revision, published_at FROM active_event WHERE singleton_id = 1"
            ).fetchone()
            if active is not None:
                profile_id, revision, published_at = str(active[0]), int(active[1]), active[2]
                document = profiles_by_revision.get((profile_id, revision))
                if document is None:
                    raise RuntimeError("The active Event revision is missing")
                snapshot = routing_snapshot(document, revision)
                database.execute(
                    "INSERT INTO active_routing_profile "
                    "(singleton_id, profile_id, revision, routing_json, published_at) VALUES (1, ?, ?, ?, ?)",
                    (profile_id, revision, json.dumps(snapshot, sort_keys=True), published_at),
                )
            database.execute("DROP TABLE active_event")
            database.execute("DROP TABLE event_revisions")
            database.execute("DROP TABLE events")
            database.execute("UPDATE schema_metadata SET value = '3' WHERE key = 'schema_version'")
            database.commit()
        except Exception:
            database.rollback()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a local ModelDeck v2 database to v3.")
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    migrate(args.database)


if __name__ == "__main__":
    main()
