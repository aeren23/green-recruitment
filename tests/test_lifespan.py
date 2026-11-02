"""The embedding model is loaded once, in the lifespan, never in a request."""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.main import create_app
from conftest import FakeEmbedder

HEALTH_REQUESTS = 3


class LoadCountingEmbedder(FakeEmbedder):
    """A fake whose `load()` calls are counted."""

    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0

    def load(self) -> None:
        """Count the call."""
        self.load_calls += 1


def test_injected_embedder_is_used_as_is_and_never_loaded(
    test_settings: Settings,
) -> None:
    """The lifespan stores the injected object and does not call `load()`."""
    injected = LoadCountingEmbedder()
    application = create_app(test_settings, injected)
    with TestClient(application) as client:
        for _ in range(HEALTH_REQUESTS):
            assert client.get("/health").status_code == 200
        assert application.state.embedder is injected
    assert injected.load_calls == 0


def test_without_injection_the_model_loads_exactly_once_at_startup(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One load during start-up, none during requests; the object is on the state."""
    stub = LoadCountingEmbedder()
    monkeypatch.setattr(main_module, "create_embedder", lambda settings: stub)
    application = create_app(test_settings)
    with TestClient(application) as client:
        assert stub.load_calls == 1
        for _ in range(HEALTH_REQUESTS):
            assert client.get("/health").status_code == 200
        assert application.state.embedder is stub
    assert stub.load_calls == 1


def test_building_the_app_does_not_load_anything(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading belongs to start-up: `create_app` alone builds no embedder."""
    stub = LoadCountingEmbedder()
    monkeypatch.setattr(main_module, "create_embedder", lambda settings: stub)
    create_app(test_settings)
    assert stub.load_calls == 0
