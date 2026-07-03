import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from aviation_rag.runtime import (
    build_answer_service,
    build_hybrid_retriever,
    load_generation_config,
    load_retrieval_config,
    build_runtime_answer_service,
)

from aviation_rag.embedding_artifacts import (
    save_embedding_artifact,
)

from types import SimpleNamespace


def write_config(
    path: Path,
    prompt: str = "Answer only from evidence.",
) -> None:
    config = {
        "prompt_version": "dev_v2",
        "system_prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "system_instructions": prompt,
        "model": "gemini-test",
        "temperature": 0.0,
        "max_output_tokens": 500,
        "thinking_budget": 0,
        "retrieval_top_k": 5,
    }

    path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )


def test_load_generation_config_accepts_valid_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "generation.json"
    write_config(config_path)

    config = load_generation_config(config_path)

    assert config["prompt_version"] == "dev_v2"
    assert config["retrieval_top_k"] == 5


def test_load_generation_config_rejects_changed_prompt(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "generation.json"
    write_config(config_path)

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    config["system_instructions"] = "Modified prompt"

    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        load_generation_config(config_path)


def test_load_generation_config_rejects_missing_key(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "generation.json"
    write_config(config_path)

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )
    del config["model"]

    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="missing keys",
    ):
        load_generation_config(config_path)


class FakeSemanticModel:
    def encode_document(
        self,
        texts,
        **kwargs,
    ):
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

    def encode_query(
        self,
        texts,
        **kwargs,
    ):
        return np.array([[1.0, 0.0]])

class QueryOnlySemanticModel(FakeSemanticModel):
    """Allow query encoding but reject document re-encoding."""

    def encode_document(
        self,
        texts,
        **kwargs,
    ):
        raise AssertionError(
            "Runtime must not re-encode document chunks"
        )


def valid_retrieval_config() -> dict:
    return {
        "status": (
            "locked_retrieval_evaluation_completed"
        ),
        "semantic_retriever": {
            "query_instruction": "Find evidence: ",
        },
        "hybrid_retriever": {
            "candidate_k": 2,
            "rrf_constant": 60,
            "lexical_weight": 0.5,
        },
    }


def test_load_retrieval_config_accepts_frozen_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "retrieval.json"
    config_path.write_text(
        json.dumps(valid_retrieval_config()),
        encoding="utf-8",
    )

    config = load_retrieval_config(config_path)

    assert config["hybrid_retriever"][
        "candidate_k"
    ] == 2


def test_load_retrieval_config_rejects_unfinished_status(
    tmp_path: Path,
) -> None:
    config = valid_retrieval_config()
    config["status"] = "development"

    config_path = tmp_path / "retrieval.json"
    config_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="locked evaluation",
    ):
        load_retrieval_config(config_path)


def test_build_hybrid_retriever_returns_results() -> None:
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "document_id": "document_1",
                "page_number": 1,
                "chunk_text": "refund passenger ticket",
            },
            {
                "chunk_id": "chunk_2",
                "document_id": "document_2",
                "page_number": 2,
                "chunk_text": "airport service report",
            },
        ]
    )

    retriever = build_hybrid_retriever(
        chunks=chunks,
        config=valid_retrieval_config(),
        semantic_model=FakeSemanticModel(),
    )

    results = retriever.retrieve(
        query="refund ticket",
        top_k=1,
    )

    assert len(results) == 1
    assert results.iloc[0]["chunk_id"] == "chunk_1"

def test_build_answer_service_connects_pipeline() -> None:
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int):
            return pd.DataFrame(
                [
                    {
                        "document_id": "document_1",
                        "page_number": 4,
                        "chunk_text": "The rule started in 2016.",
                    }
                ]
            )

    def fake_generator(messages):
        assert "What is the rule?" in messages[1]["content"]

        return {
            "answer": (
                "The rule started in 2016 "
                "[document_1, p. 4]."
            )
        }

    generation_config = {
        "system_instructions": (
            "Answer only from supplied evidence."
        ),
        "retrieval_top_k": 5,
    }

    answer_service = build_answer_service(
        retriever=FakeRetriever(),
        generator=fake_generator,
        generation_config=generation_config,
    )

    result = answer_service("What is the rule?")

    assert result["answer"].startswith(
        "The rule started in 2016"
    )
    assert result["citation_check"][
        "all_citations_valid"
    ] is True

def test_build_runtime_answer_service_without_real_api(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    retrieval_path = tmp_path / "retrieval.json"
    generation_path = tmp_path / "generation.json"
    embeddings_path = tmp_path / "embeddings.npy"
    embedding_metadata_path = tmp_path / "embeddings.json"

    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "chunk_1",
                "document_id": "document_1",
                "page_number": 1,
                "chunk_text": "The rule started in 2016.",
            },
            {
                "chunk_id": "chunk_2",
                "document_id": "document_2",
                "page_number": 2,
                "chunk_text": "Airport service report.",
            },
        ]
    )

    chunks.to_json(
        chunks_path,
        orient="records",
        lines=True,
    )

    save_embedding_artifact(
    embeddings=np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    chunks=chunks,
    model_name="fake-bge",
    embeddings_path=embeddings_path,
    metadata_path=embedding_metadata_path,
    )

    retrieval_config = valid_retrieval_config()
    retrieval_config["semantic_retriever"][
        "model"
    ] = "fake-bge"

    retrieval_path.write_text(
        json.dumps(retrieval_config),
        encoding="utf-8",
    )

    write_config(generation_path)

    class FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                raise RuntimeError("503 UNAVAILABLE")

            return SimpleNamespace(
                text=(
                    "The rule started in 2016 "
                    "[document_1, p. 1]."
                ),
                candidates=[
                    SimpleNamespace(
                        finish_reason="STOP"
                    )
                ],
                usage_metadata=None,
            )

    fake_models = FakeModels()

    fake_client = SimpleNamespace(
        models=fake_models
    )

    answer_service = build_runtime_answer_service(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        embedding_metadata_path=embedding_metadata_path,
        retrieval_config_path=retrieval_path,
        generation_config_path=generation_path,
        gemini_client=fake_client,
        semantic_model_loader=lambda name: (
            QueryOnlySemanticModel()
        ),
        retry_base_wait_seconds=0,
    )

    result = answer_service("When did the rule start?")

    assert fake_models.calls == 2
    assert result["citation_check"][
        "all_citations_valid"
    ] is True