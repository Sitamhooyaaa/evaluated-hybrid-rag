import pandas as pd
import pytest

from aviation_rag.evaluation_data import (
    evaluate_retriever,
    validate_gold_evidence,
)


def make_answerable_question() -> dict:
    """Create a minimal answerable question for testing."""

    return {
        "question_id": "DEV_TEST_001",
        "answerable": True,
        "gold_document_ids": ["document_a"],
        "gold_pages": {"document_a": [2]},
    }


def test_valid_gold_evidence_returns_no_errors() -> None:
    questions = [make_answerable_question()]

    pages = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 2,
                "retrieval_eligible": True,
            }
        ]
    )

    errors = validate_gold_evidence(questions, pages)

    assert errors == []


def test_excluded_gold_page_returns_error() -> None:
    questions = [make_answerable_question()]

    pages = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 2,
                "retrieval_eligible": False,
            }
        ]
    )

    errors = validate_gold_evidence(questions, pages)

    assert errors == [
        "DEV_TEST_001: page is excluded from retrieval: ('document_a', 2)"
    ]

def test_missing_gold_page_returns_error() -> None:
    questions = [make_answerable_question()]

    pages = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 1,
                "retrieval_eligible": True,
            }
        ]
    )

    errors = validate_gold_evidence(questions, pages)

    assert errors == [
        "DEV_TEST_001: page does not exist: ('document_a', 2)"
    ]


def test_refusal_question_with_gold_evidence_returns_error() -> None:
    questions = [
        {
            "question_id": "DEV_REFUSE_TEST",
            "answerable": False,
            "gold_document_ids": ["document_a"],
            "gold_pages": {"document_a": [2]},
        }
    ]

    pages = pd.DataFrame(
        columns=["document_id", "page_number", "retrieval_eligible"]
    )

    errors = validate_gold_evidence(questions, pages)

    assert errors == [
        "DEV_REFUSE_TEST: refusal question should not have gold evidence"
    ]


def test_mismatched_gold_document_fields_returns_error() -> None:
    questions = [
        {
            "question_id": "DEV_TEST_002",
            "answerable": True,
            "gold_document_ids": ["document_a"],
            "gold_pages": {"document_b": [2]},
        }
    ]

    pages = pd.DataFrame(
        [
            {
                "document_id": "document_b",
                "page_number": 2,
                "retrieval_eligible": True,
            }
        ]
    )

    errors = validate_gold_evidence(questions, pages)

    assert errors == [
        "DEV_TEST_002: gold document fields do not match"
    ]

class StaticRetriever:
    """Return a fixed ranking for evaluation tests."""

    def __init__(self, results: pd.DataFrame) -> None:
        self.results = results
        self.requested_top_k = None

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> pd.DataFrame:
        self.requested_top_k = top_k

        return self.results.head(top_k).copy()


def test_evaluate_retriever_calculates_multi_page_recall() -> None:
    questions = [
        {
            "question_id": "DEV_MULTI_001",
            "question": "Test question",
            "category": "cross_document_synthesis",
            "answerable": True,
            "gold_pages": {
                "document_a": [1, 2],
            },
        }
    ]

    ranked_results = pd.DataFrame(
        [
            {
                "document_id": "document_a",
                "page_number": 1,
            },
            {
                "document_id": "document_x",
                "page_number": 9,
            },
            {
                "document_id": "document_a",
                "page_number": 2,
            },
        ]
    )

    retriever = StaticRetriever(ranked_results)

    evaluation = evaluate_retriever(
        questions=questions,
        retriever=retriever,
        top_k_values=(1, 3),
    )

    assert evaluation.iloc[0]["recall_at_1"] == 0.5
    assert evaluation.iloc[0]["hit_at_1"] == 1
    assert evaluation.iloc[0]["recall_at_3"] == 1.0
    assert retriever.requested_top_k == 3


def test_evaluate_retriever_excludes_refusal_questions() -> None:
    questions = [
        {
            "question_id": "DEV_REFUSE_001",
            "question": "Unsupported question",
            "category": "refusal",
            "answerable": False,
            "gold_pages": {},
        }
    ]

    retriever = StaticRetriever(pd.DataFrame())

    evaluation = evaluate_retriever(
        questions=questions,
        retriever=retriever,
    )

    assert evaluation.empty


@pytest.mark.parametrize(
    "top_k_values",
    [
        (),
        (0, 1),
    ],
)
def test_evaluate_retriever_rejects_invalid_k_values(
    top_k_values: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        evaluate_retriever(
            questions=[],
            retriever=StaticRetriever(pd.DataFrame()),
            top_k_values=top_k_values,
        )