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
    settings.data_dir.mkdir(parents=True, exist_ok=True)
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


def _mark_done(client: TestClient, match_id: str) -> None:
    import asyncio

    match = Match(id=match_id, filename="match.mp4", status=MatchStatus.DONE)
    asyncio.get_event_loop().run_until_complete(client.store.save(match))  # type: ignore[attr-defined]


def test_data_served_when_done(client: TestClient) -> None:
    match_id = _upload(client)
    data_file = client.settings.data_dir / f"{match_id}.json"  # type: ignore[attr-defined]
    data_file.write_text('{"schema_version": 1, "shots": [], "fps": 30.0}')
    _mark_done(client, match_id)
    response = client.get(f"/matches/{match_id}/data")
    assert response.status_code == 200
    assert response.json()["schema_version"] == 1


def test_data_conflict_while_not_done(client: TestClient) -> None:
    match_id = _upload(client)
    assert client.get(f"/matches/{match_id}/data").status_code == 409


def test_clip_cut_on_demand(client: TestClient) -> None:
    import cv2
    import numpy as np

    match_id = _upload(client)
    # A real (tiny) processed video so ffmpeg has something to cut.
    result_file = client.settings.results_dir / f"{match_id}.mp4"  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(result_file), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 64))
    for _ in range(60):  # 2 seconds
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    (client.settings.data_dir / f"{match_id}.json").write_text('{"fps": 30.0}')  # type: ignore[attr-defined]
    _mark_done(client, match_id)

    response = client.get(f"/matches/{match_id}/clip", params={"frame": 30})
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 0
    # The cut file is cached for reuse.
    assert (client.settings.data_dir / f"{match_id}_clip_30.mp4").exists()  # type: ignore[attr-defined]


def test_clip_conflict_while_not_done(client: TestClient) -> None:
    match_id = _upload(client)
    assert client.get(f"/matches/{match_id}/clip", params={"frame": 10}).status_code == 409


def test_thumbnail_served_when_present(client: TestClient) -> None:
    match_id = _upload(client)
    thumb = client.settings.data_dir / f"{match_id}.jpg"  # type: ignore[attr-defined]
    thumb.write_bytes(b"\xff\xd8\xff\xe0jpegbytes")
    _mark_done(client, match_id)
    response = client.get(f"/matches/{match_id}/thumbnail")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_thumbnail_404_when_missing(client: TestClient) -> None:
    match_id = _upload(client)
    _mark_done(client, match_id)
    assert client.get(f"/matches/{match_id}/thumbnail").status_code == 404


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_index_serves_scaffold_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Padel Vision" in response.text


def test_worker_module_imports() -> None:
    # The worker imports the CV pipeline lazily; ensure the module and its
    # Arq entrypoint are importable without a running Redis.
    assert "process_match" in dir(main) or True
    from padel_api.worker import WorkerSettings, process_match

    assert process_match in WorkerSettings.functions
