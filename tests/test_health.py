"""The health endpoint works through the real lifespan."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from conftest import FakeEmbedder


def test_health_returns_ok(
    test_settings: Settings, fake_embedder: FakeEmbedder
) -> None:
    """GET /health answers 200 and the lifespan wires the engine."""
    application = create_app(test_settings, fake_embedder)
    with TestClient(application) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert application.state.engine is not None
