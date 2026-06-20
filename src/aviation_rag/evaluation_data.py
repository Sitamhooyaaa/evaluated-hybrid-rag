"""Utilities for validating RAG evaluation questions."""

from typing import Any

import pandas as pd


def validate_gold_evidence(
    questions: list[dict[str, Any]],
    pages: pd.DataFrame,
) -> list[str]:
    """Return validation errors found in question gold evidence."""

    available_pages = set(
        zip(pages["document_id"], pages["page_number"])
    )

    eligible_pages = set(
        zip(
            pages.loc[pages["retrieval_eligible"], "document_id"],
            pages.loc[pages["retrieval_eligible"], "page_number"],
        )
    )

    errors = []

    for item in questions:
        question_id = item["question_id"]
        gold_document_ids = set(item["gold_document_ids"])
        gold_page_documents = set(item["gold_pages"])

        if item["answerable"]:
            if not gold_document_ids or not item["gold_pages"]:
                errors.append(
                    f"{question_id}: answerable question has no gold evidence"
                )

            if gold_document_ids != gold_page_documents:
                errors.append(
                    f"{question_id}: gold document fields do not match"
                )

            for document_id, page_numbers in item["gold_pages"].items():
                for page_number in page_numbers:
                    page_key = (document_id, page_number)

                    if page_key not in available_pages:
                        errors.append(
                            f"{question_id}: page does not exist: {page_key}"
                        )
                    elif page_key not in eligible_pages:
                        errors.append(
                            f"{question_id}: page is excluded from retrieval: {page_key}"
                        )

        elif gold_document_ids or item["gold_pages"]:
            errors.append(
                f"{question_id}: refusal question should not have gold evidence"
            )

    return errors

def evaluate_retriever(
    questions: list[dict[str, Any]],
    retriever: Any,
    top_k_values: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Evaluate ranked retrieval against page-level gold evidence."""

    if not top_k_values:
        raise ValueError("top_k_values cannot be empty")

    if any(k <= 0 for k in top_k_values):
        raise ValueError(
            "All top_k_values must be greater than zero"
        )

    maximum_k = max(top_k_values)
    evaluation_records = []

    for item in questions:
        if not item["answerable"]:
            continue

        gold_pages = {
            (document_id, int(page_number))
            for document_id, page_numbers
            in item["gold_pages"].items()
            for page_number in page_numbers
        }

        retrieved = retriever.retrieve(
            query=item["question"],
            top_k=maximum_k,
        )

        retrieved_pages = list(
            zip(
                retrieved["document_id"],
                retrieved["page_number"].astype(int),
            )
        )

        record = {
            "question_id": item["question_id"],
            "category": item["category"],
            "gold_page_count": len(gold_pages),
        }

        for k in top_k_values:
            top_k_pages = set(retrieved_pages[:k])
            matched_pages = gold_pages & top_k_pages

            record[f"recall_at_{k}"] = (
                len(matched_pages) / len(gold_pages)
            )

            record[f"hit_at_{k}"] = int(
                bool(matched_pages)
            )

        evaluation_records.append(record)

    return pd.DataFrame(evaluation_records)