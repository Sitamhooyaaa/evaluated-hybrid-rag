from fastapi.testclient import TestClient

from aviation_rag.api import create_app
from aviation_rag.gemini import GeminiServiceError
import json
import logging


def test_health_check_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_grounded_answer() -> None:
    def fake_answer_service(question: str) -> dict:
        return {
            "answer": "The Code started on 1 July 2016.",
            "citation_check": {
                "cited_sources": [
                    ("macpc_principal_2016", 39)
                ]
            },
        }

    client = TestClient(
        create_app(
            answer_service=fake_answer_service,
        )
    )

    response = client.post(
        "/ask",
        json={"question": "When did the Code start?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "When did the Code start?",
        "answer": "The Code started on 1 July 2016.",
        "citations": [
            {
                "document_id": "macpc_principal_2016",
                "page_number": 39,
            }
        ],
    }


def test_ask_without_service_returns_503() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        json={"question": "When did the Code start?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Answer service is not configured"
    }


def test_ask_rejects_blank_question() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422

def test_ask_rejects_invalid_generated_citation() -> None:
    def fake_answer_service(question: str) -> dict:
        return {
            "answer": "Unsupported claim [fake_doc, p. 99].",
            "citation_check": {
                "cited_sources": [
                    ("fake_doc", 99)
                ],
                "invalid_citations": [
                    ("fake_doc", 99)
                ],
                "malformed_citations": [],
            },
        }

    client = TestClient(
        create_app(
            answer_service=fake_answer_service,
        )
    )

    response = client.post(
        "/ask",
        json={"question": "What is the rule?"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": (
            "Generated answer failed citation validation"
        )
    }

def test_ask_returns_503_when_gemini_fails() -> None:
    def failing_answer_service(question: str) -> dict:
        raise GeminiServiceError(
            "Gemini request failed: 503 UNAVAILABLE"
        )

    client = TestClient(
        create_app(
            answer_service=failing_answer_service,
        )
    )

    response = client.post(
        "/ask",
        json={"question": "What is the rule?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Generation service is temporarily unavailable"
        )
    }

def test_request_logging_excludes_question_text(
    caplog,
) -> None:
    secret_question = (
        "My private booking reference is ABC123"
    )

    def fake_answer_service(question: str) -> dict:
        return {
            "answer": "Test answer.",
            "citation_check": {
                "cited_sources": [],
                "invalid_citations": [],
                "malformed_citations": [],
            },
        }

    client = TestClient(
        create_app(
            answer_service=fake_answer_service,
        )
    )

    with caplog.at_level(
        logging.INFO,
        logger="uvicorn.error",
    ):
        response = client.post(
            "/ask",
            json={"question": secret_question},
        )

    request_logs = [
        json.loads(record.message)
        for record in caplog.records
        if '"event":"http_request_completed"'
        in record.message
    ]

    assert response.status_code == 200
    assert len(request_logs) == 1

    request_log = request_logs[0]

    assert request_log["method"] == "POST"
    assert request_log["path"] == "/ask"
    assert request_log["status_code"] == 200
    assert request_log["duration_ms"] >= 0
    assert request_log["request_id"]
    assert response.headers["X-Request-ID"] == (
        request_log["request_id"]
    )
    assert secret_question not in caplog.text