from __future__ import annotations

import json
import sqlite3
from uuid import uuid4

from modeldeck.compatibility import CompatibilityStore
from modeldeck.migrate_v3_to_v4 import migrate


def test_migration_grandfathers_current_workers_without_historical_routes(tmp_path) -> None:
    path = tmp_path / "modeldeck.sqlite3"
    current_worker_id = str(uuid4())
    archived_worker_id = str(uuid4())
    profile_id = str(uuid4())
    current_worker = {
        "id": current_worker_id,
        "name": "Current Worker",
        "model_id": "Qwen/current",
        "revision": "current-revision",
        "generation_family": "autoregressive",
        "runtime": "transformers-rocm",
        "runtime_template_id": "autoregressive-transformers",
        "runtime_template_version": "1",
        "lifecycle": "on-demand",
        "port": 8610,
        "dtype": "float16",
        "capabilities": {"chat": True, "completions": True, "top_k_trace": True},
        "settings": {},
    }
    archived_worker = {
        **current_worker,
        "id": archived_worker_id,
        "name": "Archived Worker",
        "model_id": "google/archived-diffusion",
        "revision": "archived-revision",
        "generation_family": "text-diffusion",
        "runtime": "text-diffusion-transformers-rocm",
        "runtime_template_id": "diffusiongemma-transformers",
        "capabilities": {"iterative_refinement": True, "intermediate_frames": True},
        "port": 8611,
    }
    draft = {
        "id": profile_id,
        "name": "Current profile",
        "description": "",
        "qualification": "compatible",
        "capabilities": [
            {
                "id": str(uuid4()),
                "display_name": "Chat",
                "public_name": "current-chat",
                "protocol_contract": "openai-chat-v1",
                "worker_ids": [current_worker_id],
            }
        ],
    }
    history = {
        **draft,
        "capabilities": [
            {
                "id": str(uuid4()),
                "display_name": "Old refinement",
                "public_name": "old-refinement",
                "protocol_contract": "text-diffusion-v1",
                "worker_ids": [archived_worker_id],
            }
        ],
    }
    with sqlite3.connect(path) as database:
        database.executescript(
            """
            CREATE TABLE schema_metadata (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE workers (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, document_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, archived_at TEXT
            );
            CREATE TABLE routing_profiles (
                id TEXT PRIMARY KEY, draft_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE routing_profile_revisions (
                profile_id TEXT NOT NULL, revision INTEGER NOT NULL,
                document_json TEXT NOT NULL, published_at TEXT NOT NULL,
                PRIMARY KEY (profile_id, revision)
            );
            """
        )
        database.execute("INSERT INTO schema_metadata VALUES ('schema_version', '3', 'now')")
        database.execute(
            "INSERT INTO workers VALUES (?, ?, ?, 'now', 'now', NULL)",
            (current_worker_id, current_worker["name"], json.dumps(current_worker)),
        )
        database.execute(
            "INSERT INTO workers VALUES (?, ?, ?, 'now', 'now', 'now')",
            (archived_worker_id, archived_worker["name"], json.dumps(archived_worker)),
        )
        database.execute(
            "INSERT INTO routing_profiles VALUES (?, ?, 'now', 'now')",
            (profile_id, json.dumps(draft)),
        )
        database.execute(
            "INSERT INTO routing_profile_revisions VALUES (?, 1, ?, 'now')",
            (profile_id, json.dumps(history)),
        )

    migrate(path)
    migrate(path)
    store = CompatibilityStore(path)
    store.initialise_v4()

    assert store.model_capability_allowed("Qwen/current", "current-revision", "general-chat")
    assert not store.model_capability_allowed(
        "google/archived-diffusion", "archived-revision", "text-refinement"
    )
