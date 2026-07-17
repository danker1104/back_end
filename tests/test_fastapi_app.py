import builtins
import importlib
import sys

from fastapi.testclient import TestClient


def test_app_import_is_lazy_for_chat_dependency(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "chat":
            raise ImportError("chat unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("app", None)

    module = importlib.import_module("app")

    assert module.app is not None


def test_health_endpoint_returns_ok():
    module = importlib.import_module("app")
    client = TestClient(module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
