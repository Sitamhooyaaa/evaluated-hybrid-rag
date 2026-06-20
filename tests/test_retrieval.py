import pandas as pd
import pytest
import numpy as np

from aviation_rag.retrieval import (
    HybridRetriever,
    SemanticRetriever,
    TfidfRetriever,
)


def make_chunks() -> pd.DataFrame:
    """Create a small retrieval corpus for testing."""

    return pd.DataFrame(
        [
            {
                "chunk_id": "chunk_a",
                "document_id": "document_a",
                "page_number": 1,
                "chunk_text": (
                    "Passengers affected by flight cancellation "
                    "may request a refund."
                ),
            },
            {
                "chunk_id": "chunk_b",
                "document_id": "document_b",
                "page_number": 2,
                "chunk_text": (
                    "The report contains airport baggage complaints."
                ),
            },
        ]
    )


def test_tfidf_ranks_matching_chunk_first() -> None:
    retriever = TfidfRetriever(make_chunks())

    results = retriever.retrieve(
        "flight cancellation refund",
        top_k=2,
    )

    assert results.iloc[0]["chunk_id"] == "chunk_a"
    assert results.iloc[0]["score"] > results.iloc[1]["score"]


def test_tfidf_limits_result_count_to_corpus_size() -> None:
    retriever = TfidfRetriever(make_chunks())

    results = retriever.retrieve(
        "passenger",
        top_k=10,
    )

    assert len(results) == 2


def test_tfidf_rejects_invalid_top_k() -> None:
    retriever = TfidfRetriever(make_chunks())

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        retriever.retrieve("passenger", top_k=0)


def test_tfidf_rejects_missing_chunk_columns() -> None:
    incomplete_chunks = pd.DataFrame(
        [{"chunk_id": "chunk_a"}]
    )

    with pytest.raises(
        ValueError,
        match="Chunk dataset is missing columns",
    ):
        TfidfRetriever(incomplete_chunks)


def test_tfidf_rejects_empty_chunk_dataset() -> None:
    empty_chunks = pd.DataFrame(
        columns=[
            "chunk_id",
            "document_id",
            "page_number",
            "chunk_text",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Chunk dataset cannot be empty",
    ):
        TfidfRetriever(empty_chunks)

class FakeEmbeddingModel:
    """Return deterministic vectors without downloading a model."""

    def __init__(self) -> None:
        self.last_query = None

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        if "flight" in text.lower():
            return [1.0, 0.0]

        return [0.0, 1.0]

    def encode_document(
        self,
        texts,
        **kwargs,
    ) -> np.ndarray:
        return np.array(
            [self._vectorize(text) for text in texts]
        )

    def encode_query(
        self,
        queries,
        **kwargs,
    ) -> np.ndarray:
        self.last_query = queries[0]

        return np.array(
            [self._vectorize(query) for query in queries]
        )


def test_semantic_retriever_ranks_matching_chunk_first() -> None:
    model = FakeEmbeddingModel()

    retriever = SemanticRetriever(
        chunks=make_chunks(),
        model=model,
    )

    results = retriever.retrieve(
        "flight cancellation",
        top_k=2,
    )

    assert results.iloc[0]["chunk_id"] == "chunk_a"
    assert results.iloc[0]["score"] > results.iloc[1]["score"]


def test_semantic_retriever_applies_query_prefix() -> None:
    model = FakeEmbeddingModel()

    retriever = SemanticRetriever(
        chunks=make_chunks(),
        model=model,
        query_prefix="search: ",
    )

    retriever.retrieve("flight cancellation")

    assert model.last_query == "search: flight cancellation"

def make_component_retrievers():
    chunks = make_chunks()

    lexical_retriever = TfidfRetriever(chunks)

    semantic_retriever = SemanticRetriever(
        chunks=chunks,
        model=FakeEmbeddingModel(),
    )

    return lexical_retriever, semantic_retriever


def test_hybrid_retriever_ranks_shared_match_first() -> None:
    lexical_retriever, semantic_retriever = (
        make_component_retrievers()
    )

    retriever = HybridRetriever(
        lexical_retriever=lexical_retriever,
        semantic_retriever=semantic_retriever,
    )

    results = retriever.retrieve(
        "flight cancellation",
        top_k=2,
    )

    assert results.iloc[0]["chunk_id"] == "chunk_a"
    assert len(results) == 2


def test_hybrid_retriever_rejects_different_datasets() -> None:
    lexical_retriever = TfidfRetriever(make_chunks())

    different_chunks = make_chunks().copy()
    different_chunks.loc[0, "chunk_id"] = "different_chunk"

    semantic_retriever = SemanticRetriever(
        chunks=different_chunks,
        model=FakeEmbeddingModel(),
    )

    with pytest.raises(
        ValueError,
        match="Retrievers must use the same chunk dataset",
    ):
        HybridRetriever(
            lexical_retriever=lexical_retriever,
            semantic_retriever=semantic_retriever,
        )


@pytest.mark.parametrize(
    "invalid_parameters",
    [
        {"candidate_k": 0},
        {"rrf_constant": 0},
        {"lexical_weight": 1.1},
    ],
)
def test_hybrid_retriever_rejects_invalid_parameters(
    invalid_parameters: dict,
) -> None:
    lexical_retriever, semantic_retriever = (
        make_component_retrievers()
    )

    with pytest.raises(ValueError):
        HybridRetriever(
            lexical_retriever=lexical_retriever,
            semantic_retriever=semantic_retriever,
            **invalid_parameters,
        )