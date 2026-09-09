from fastapi.testclient import TestClient

from app.api.routes import llm_service, stt_service
from app.main import app


def test_root(monkeypatch):
    monkeypatch.setattr(stt_service, "load", lambda: None)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health(monkeypatch):
    monkeypatch.setattr(stt_service, "load", lambda: None)
    stt_service.model = object()
    monkeypatch.setattr(llm_service, "health", lambda: True)

    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["stt_loaded"] is True
        assert payload["ollama_reachable"] is True
    finally:
        stt_service.model = None
