"""Testes da API (unitários com tradutor falso + integração opcional com o modelo real).

Integração:  RUN_INTEGRATION=1 pytest
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_translator
from app.translator import TranslationResult


class FakeTranslator:
    model_name = "fake-opus-mt-en-fr"

    def translate(self, text: str) -> TranslationResult:
        return TranslationResult(text=f"[fr] {text}", chunks=1)


@pytest.fixture
def client():
    app.dependency_overrides[get_translator] = lambda: FakeTranslator()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_translate_returns_json(client):
    response = client.post("/translate", json={"text": "Hello"})
    assert response.status_code == 200
    assert response.json() | {"elapsed_ms": 0} == {
        "source_language": "en",
        "target_language": "fr",
        "original_text": "Hello",
        "translated_text": "[fr] Hello",
        "model": "fake-opus-mt-en-fr",
        "chunks": 1,
        "elapsed_ms": 0,
    }


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "a" * 10_001}])
def test_translate_rejects_invalid_text(client, payload):
    assert client.post("/translate", json=payload).status_code == 422


@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="defina RUN_INTEGRATION=1")
def test_translate_with_real_model_short_and_long_text():
    long_text = "\n\n".join(["The weather is nice today, so we decided to go for a walk in the park. " * 15] * 3)
    with TestClient(app) as real_client:
        short = real_client.post("/translate", json={"text": "I love programming."}).json()
        long = real_client.post("/translate", json={"text": long_text}).json()

    assert "programm" in short["translated_text"].lower()  # "J'adore programmer."
    assert long["chunks"] >= 3
    assert long["translated_text"].count("\n\n") == 2  # 3 parágrafos preservados
