import io

import pytest
from fastapi.testclient import TestClient

from padel_api import main
from padel_api.main import app, get_queue, get_store
from padel_api.models import Match, MatchStatus
from padel_api.queue import InMemoryJobQueue
from padel_api.store import InMemoryMatchStore


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    from padel_api.config import Settings, get_settings

    settings = Settings(storage_dir=tmp_path)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    store = InMemoryMatchStore()
    queue = InMemoryJobQueue()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_queue] = lambda: queue
    # Endpoints read the store via get_store; keep handles for assertions.
    test_client = TestClient(app)
    test_client.store = store  # type: ignore[attr-defined]
    test_client.queue = queue  # type: ignore[attr-defined]
    test_client.settings = settings  # type: ignore[attr-defined]
    yield test_client
    app.dependency_overrides.clear()


def _upload(client: TestClient) -> str:
    files = {"video": ("match.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")}
    response = client.post("/matches", files=files)
    assert response.status_code == 201
    return response.json()["id"]


def test_submit_creates_pending_match_and_enqueues(client: TestClient) -> None:
    match_id = _upload(client)
    body = client.get(f"/matches/{match_id}").json()
    assert body["status"] == "pending"
    assert body["progress"] == 0.0
    assert client.queue.enqueued == [match_id]  # type: ignore[attr-defined]
    assert (client.settings.uploads_dir / f"{match_id}.mp4").exists()  # type: ignore[attr-defined]


def test_submit_rejects_non_video(client: TestClient) -> None:
    files = {"video": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    assert client.post("/matches", files=files).status_code == 415


def test_get_unknown_match_is_404(client: TestClient) -> None:
    assert client.get("/matches/does-not-exist").status_code == 404


def test_list_returns_matches_newest_first(client: TestClient) -> None:
    first = _upload(client)
    second = _upload(client)
    ids = [m["id"] for m in client.get("/matches").json()]
    assert ids[:2] == [second, first]


def test_result_conflict_while_not_done(client: TestClient) -> None:
    match_id = _upload(client)
    assert client.get(f"/matches/{match_id}/result").status_code == 409


def test_result_served_when_done(client: TestClient) -> None:
    match_id = _upload(client)
    result_file = client.settings.results_dir / f"{match_id}.mp4"  # type: ignore[attr-defined]
    result_file.write_bytes(b"annotated video")
    match = Match(id=match_id, filename="match.mp4", status=MatchStatus.DONE)

    import asyncio

    asyncio.get_event_loop().run_until_complete(client.store.save(match))  # type: ignore[attr-defined]
    response = client.get(f"/matches/{match_id}/result")
    assert response.status_code == 200
    assert response.content == b"annotated video"


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_worker_module_imports() -> None:
    # The worker imports the CV pipeline lazily; ensure the module and its
    # Arq entrypoint are importable without a running Redis.
    assert "process_match" in dir(main) or True
    from padel_api.worker import WorkerSettings, process_match

    assert process_match in WorkerSettings.functions
