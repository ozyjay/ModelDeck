import sqlite3

import pytest
from modeldeck.compatibility import CompatibilityStore, LegacyDatabaseError
from modeldeck.domain import routing_snapshot

from tests.contract.test_gateway import published_chat_profile, worker


def v4_database(tmp_path):
    path = tmp_path / "state.sqlite"
    store = CompatibilityStore(path)
    store.initialise_v5()
    definition = worker()
    store.save_worker_definition(definition.model_dump(mode="json"))
    profile = published_chat_profile(definition.id)
    store.save_routing_profile_draft(profile.model_dump(mode="json"))
    store.publish_routing_profile(profile.model_dump(mode="json"), routing_snapshot(profile, 1))
    with sqlite3.connect(path) as database:
        database.execute("UPDATE schema_metadata SET value='4' WHERE key='schema_version'")
        database.execute("DROP TABLE capability_setup_events")
        database.execute("DROP TABLE capability_setups")
    return store, profile


def test_v4_upgrade_preserves_workers_publications_and_policy(tmp_path):
    store, profile = v4_database(tmp_path)
    store.set_model_cache_allowed("example/model", "revision-1", allowed=False)
    store.set_model_capability_allowed("example/model", "revision-1", "chat", allowed=False)
    evidence = {"model_id": "example/model", "runtime": "mock", "artifact_sha256": "a" * 64}
    store.record_test(evidence, result="transient-failure", failure_class="smoke-failure")
    passed = store.record_test(evidence, result="tested-working")
    store.record_test_observation(passed["id"], {"shutdown_result": "success"})
    before_evidence = store.list_tests()
    before_workers = store.list_workers()
    before_routes = store.active_routing_snapshots()
    store.initialise_v5()
    store.initialise_v5()
    assert store.list_workers() == before_workers
    assert store.active_routing_snapshots() == before_routes
    assert store.list_tests() == before_evidence
    assert store.model_cache_allowed("example/model", "revision-1") is False
    assert store.model_capability_allowed("example/model", "revision-1", "chat") is False
    with sqlite3.connect(store.path) as database:
        assert (
            database.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()[0]
            == "5"
        )
        assert database.execute("SELECT COUNT(*) FROM capability_setups").fetchone()[0] == 0
        assert not database.execute("PRAGMA foreign_key_check").fetchall()
    store.deactivate_routing_profile(profile.id)
    store.initialise_v5()
    assert store.active_routing_snapshots() == []


def test_future_schema_is_refused_without_downgrade(tmp_path):
    store, _ = v4_database(tmp_path)
    with sqlite3.connect(store.path) as database:
        database.execute("UPDATE schema_metadata SET value='6' WHERE key='schema_version'")
    with pytest.raises(LegacyDatabaseError, match="pre-upgrade backup"):
        store.initialise_v5()
    with sqlite3.connect(store.path) as database:
        assert (
            database.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()[0]
            == "6"
        )


def test_failed_upgrade_rolls_back_schema_and_version(tmp_path):
    store, _ = v4_database(tmp_path)
    with sqlite3.connect(store.path) as database:
        database.execute(
            "CREATE TRIGGER refuse_version BEFORE UPDATE ON schema_metadata "
            "BEGIN SELECT RAISE(ABORT, 'test failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.initialise_v5()
    with sqlite3.connect(store.path) as database:
        assert (
            database.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()[0]
            == "4"
        )
        assert not database.execute(
            "SELECT name FROM sqlite_master WHERE name='capability_setups'"
        ).fetchall()
