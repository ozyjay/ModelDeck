import sqlite3

import httpx
import pytest
from modeldeck.compatibility import CompatibilityStore
from modeldeck.config import Settings
from modeldeck.gateway import create_gateway_app
from modeldeck.main import create_app
from modeldeck.persistence import BUSY_TIMEOUT_MS, PersistenceError, connect_database


def test_factory_enforces_foreign_keys_rows_and_closes_connections(tmp_path):
    with connect_database(tmp_path / "state.sqlite", create=True) as database:
        assert database.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert database.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
        assert database.execute("SELECT 1 AS value").fetchone()["value"] == 1
        database.execute("CREATE TABLE parent(id INTEGER PRIMARY KEY)")
        database.execute("CREATE TABLE child(parent_id INTEGER REFERENCES parent(id))")
        with pytest.raises(sqlite3.IntegrityError):
            database.execute("INSERT INTO child VALUES (1)")
    with pytest.raises(sqlite3.ProgrammingError):
        database.execute("SELECT 1")


def test_missing_database_is_not_created_by_reads(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(PersistenceError, match="database_unavailable"):
        with connect_database(path):
            pass
    assert not path.exists()


def test_failed_transaction_rolls_back(tmp_path):
    path = tmp_path / "state.sqlite"
    with connect_database(path, create=True) as database:
        database.execute("CREATE TABLE values_table(value INTEGER)")
    with pytest.raises(PersistenceError):
        with connect_database(path) as database:
            database.execute("INSERT INTO values_table VALUES (1)")
            database.execute("SELECT * FROM missing_table")
    with connect_database(path) as database:
        assert database.execute("SELECT COUNT(*) FROM values_table").fetchone()[0] == 0


def test_competing_writer_reports_typed_busy_error(tmp_path, monkeypatch):
    monkeypatch.setattr("modeldeck.persistence.BUSY_TIMEOUT_MS", 20)
    path = tmp_path / "busy.sqlite"
    with connect_database(path, create=True) as database:
        database.execute("CREATE TABLE records(value INTEGER)")
    with connect_database(path) as first:
        first.execute("INSERT INTO records VALUES (1)")
        with pytest.raises(PersistenceError, match="database_busy"):
            with connect_database(path) as second:
                second.execute("INSERT INTO records VALUES (2)")


@pytest.mark.parametrize(
    "operation",
    [
        lambda store: store.list_workers(),
        lambda store: store.active_routing_snapshots(),
        lambda store: store.route_tool_calling_state("profile", 1, "capability"),
    ],
)
def test_missing_schema_is_not_empty_or_default_state(tmp_path, operation):
    path = tmp_path / "state.sqlite"
    with connect_database(path, create=True):
        pass
    with pytest.raises(PersistenceError):
        operation(CompatibilityStore(path))


@pytest.mark.parametrize("factory,path", [(create_app, "/api/health"), (create_gateway_app, "/v1/health")])
async def test_corrupt_startup_has_structured_degraded_health(tmp_path, factory, path):
    (tmp_path / "modeldeck.sqlite3").write_bytes(b"corrupt database")
    app = factory(settings=Settings(data_dir=tmp_path))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        response = await client.get(path)
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["error"]["code"] == "database_corrupt"
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize("factory,path", [(create_app, "/api/health"), (create_gateway_app, "/v1/health")])
async def test_database_removed_after_startup_reports_degraded(tmp_path, factory, path):
    app = factory(settings=Settings(data_dir=tmp_path))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        (tmp_path / "modeldeck.sqlite3").unlink()
        response = await client.get(path)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_unavailable"


@pytest.mark.parametrize("factory,path", [(create_app, "/api/health"), (create_gateway_app, "/v1/health")])
async def test_missing_operational_table_reports_degraded(tmp_path, factory, path):
    app = factory(settings=Settings(data_dir=tmp_path))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1") as client,
    ):
        with connect_database(tmp_path / "modeldeck.sqlite3") as database:
            database.execute("DROP TABLE workers")
        response = await client.get(path)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "database_schema_incomplete"
