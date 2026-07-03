from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aviation_rag.main import create_production_app


def test_production_app_builds_service_at_startup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    class FakeClient:
        def __init__(self, api_key: str):
            assert api_key == "test-key"
            self.closed = False

        def close(self):
            self.closed = True

    def fake_answer_service(question: str) -> dict:
        return {
            "answer": "Test answer [document_1, p. 1].",
            "citation_check": {
                "cited_sources": [
                    ("document_1", 1)
                ]
            },
        }

    def fake_service_builder(**kwargs):
        assert kwargs["chunks_path"] == (
            tmp_path
            / "data"
            / "processed"
            / "chunks.jsonl"
        )

        assert kwargs["embeddings_path"] == (
            tmp_path
            / "artifacts"
            / "bge_chunk_embeddings.npy"
        )

        assert kwargs["embedding_metadata_path"] == (
            tmp_path
            / "artifacts"
            / "bge_chunk_embeddings.json"
        )

        return fake_answer_service

    app = create_production_app(
        project_root=tmp_path,
        client_factory=FakeClient,
        service_builder=fake_service_builder,
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"question": "What is the rule?"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Test answer [document_1, p. 1]."
    )


def test_production_app_requires_api_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "GEMINI_API_KEY",
        raising=False,
    )

    app = create_production_app(
        project_root=tmp_path,
    )

    with pytest.raises(
        RuntimeError,
        match="GEMINI_API_KEY",
    ):
        with TestClient(app):
            pass