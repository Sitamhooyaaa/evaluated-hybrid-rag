import pandas as pd
import json

import numpy as np
import pytest

from aviation_rag.embedding_artifacts import (
    compute_chunk_dataset_sha256,
    load_embedding_artifact,
    save_embedding_artifact,
)


def make_chunks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": "chunk_a",
                "chunk_text": "First evidence passage.",
            },
            {
                "chunk_id": "chunk_b",
                "chunk_text": "Second evidence passage.",
            },
        ]
    )


def test_chunk_hash_is_stable_for_identical_data() -> None:
    chunks = make_chunks()

    first_hash = compute_chunk_dataset_sha256(chunks)
    second_hash = compute_chunk_dataset_sha256(chunks.copy())

    assert first_hash == second_hash


def test_chunk_hash_changes_when_row_order_changes() -> None:
    chunks = make_chunks()

    original_hash = compute_chunk_dataset_sha256(chunks)
    reordered_hash = compute_chunk_dataset_sha256(
        chunks.iloc[::-1].reset_index(drop=True)
    )

    assert original_hash != reordered_hash

def test_embedding_artifact_round_trip(tmp_path) -> None:
    chunks = make_chunks()

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "embeddings.json"

    save_embedding_artifact(
        embeddings=embeddings,
        chunks=chunks,
        model_name="test-model",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    loaded_embeddings = load_embedding_artifact(
        chunks=chunks,
        expected_model_name="test-model",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    assert np.array_equal(
        loaded_embeddings,
        embeddings,
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["model_name"] == "test-model"
    assert metadata["chunk_count"] == 2
    assert metadata["embedding_dimension"] == 2
    assert metadata["normalized"] is True

def test_load_rejects_different_chunk_order(
    tmp_path,
) -> None:
    chunks = make_chunks()

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "embeddings.json"

    save_embedding_artifact(
        embeddings=embeddings,
        chunks=chunks,
        model_name="test-model",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    reordered_chunks = (
        chunks.iloc[::-1]
        .reset_index(drop=True)
    )

    with pytest.raises(
        ValueError,
        match="chunk dataset does not match",
    ):
        load_embedding_artifact(
            chunks=reordered_chunks,
            expected_model_name="test-model",
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
        )

def test_load_rejects_tampered_embedding_file(
    tmp_path,
) -> None:
    chunks = make_chunks()

    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "embeddings.json"

    save_embedding_artifact(
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        chunks=chunks,
        model_name="test-model",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    embeddings_path.write_bytes(
        embeddings_path.read_bytes() + b"tampered"
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 does not match",
    ):
        load_embedding_artifact(
            chunks=chunks,
            expected_model_name="test-model",
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
        )


def test_load_rejects_wrong_embedding_model(
    tmp_path,
) -> None:
    chunks = make_chunks()

    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "embeddings.json"

    save_embedding_artifact(
        embeddings=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        chunks=chunks,
        model_name="model-a",
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )

    with pytest.raises(
        ValueError,
        match="model does not match",
    ):
        load_embedding_artifact(
            chunks=chunks,
            expected_model_name="model-b",
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
        )