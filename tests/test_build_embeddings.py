import json
from pathlib import Path

import numpy as np
import pandas as pd

from aviation_rag.build_embeddings import (
    build_embedding_artifacts,
)


class FakeEmbeddingModel:
    def encode_document(
        self,
        texts,
        **kwargs,
    ) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )


def test_build_embedding_artifacts_saves_files(
    tmp_path: Path,
) -> None:
    chunks_path = (
        tmp_path / "data" / "processed" / "chunks.jsonl"
    )
    config_path = (
        tmp_path / "evaluation" / "retrieval_config.json"
    )

    chunks_path.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)

    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "chunk_a",
                "chunk_text": "First passage.",
            },
            {
                "chunk_id": "chunk_b",
                "chunk_text": "Second passage.",
            },
        ]
    )

    chunks.to_json(
        chunks_path,
        orient="records",
        lines=True,
    )

    config_path.write_text(
        json.dumps(
            {
                "semantic_retriever": {
                    "model": "test-model",
                }
            }
        ),
        encoding="utf-8",
    )

    embeddings = build_embedding_artifacts(
        project_root=tmp_path,
        model_loader=lambda _: FakeEmbeddingModel(),
    )

    embeddings_path = (
        tmp_path
        / "artifacts"
        / "bge_chunk_embeddings.npy"
    )
    metadata_path = (
        tmp_path
        / "artifacts"
        / "bge_chunk_embeddings.json"
    )

    assert embeddings.shape == (2, 2)
    assert embeddings_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata["model_name"] == "test-model"
    assert metadata["chunk_count"] == 2