#!/usr/bin/env python
"""
seed_eval_set.py
Populate data/eval/ with a labeled starter evaluation set for AAPL.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_DIR
from app.logging_config import get_logger

logger = get_logger("seed_eval_set")

QUESTIONS = [
    {
        "id": "q001",
        "question": "What cybersecurity risks does Apple disclose in its latest 10-K?",
        "ticker": "AAPL",
        "form_type": "10-K",
        "category": "factual",
        "expected_section": "Risk Factors",
    },
    {
        "id": "q002",
        "question": "What supply chain risks does Apple highlight in its latest 10-K?",
        "ticker": "AAPL",
        "form_type": "10-K",
        "category": "risk_summary",
        "expected_section": "Risk Factors",
    },
    {
        "id": "q003",
        "question": "What macroeconomic risks does Apple identify in its latest 10-K?",
        "ticker": "AAPL",
        "form_type": "10-K",
        "category": "factual",
        "expected_section": "Risk Factors",
    },
    {
        "id": "q004",
        "question": "What changes in financial reporting standards does Apple discuss in its latest 10-K?",
        "ticker": "AAPL",
        "form_type": "10-K",
        "category": "comparative",
        "expected_section": "Management's Discussion and Analysis",
    },
    {
        "id": "q005",
        "question": "What does Apple say about adverse economic conditions in its latest 10-K?",
        "ticker": "AAPL",
        "form_type": "10-K",
        "category": "factual",
        "expected_section": "Risk Factors",
    },
    {
        "id": "q006",
        "question": "What is Apple's exact dividend yield as of last quarter?",
        "ticker": "AAPL",
        "form_type": "10-Q",
        "category": "insufficient_context",
        "expected_section": "Management's Discussion and Analysis",
        "note": "This should generally trigger an insufficient-context style answer."
    }
]

GOLD_ANSWERS = {
    "q001": {
        "reference_answer": "Apple discloses cybersecurity risks including potential data breaches, service disruptions, theft of intellectual property, and harm to its business, reputation, and financial condition.",
        "key_points": [
            "cybersecurity incidents may disrupt operations",
            "data breaches may expose sensitive information",
            "theft of intellectual property is a risk",
            "cyber incidents may harm financial condition or reputation"
        ],
        "unacceptable_claims": [
            "Apple reported a major cyberattack in the filing",
            "Apple disclosed an exact number of cybersecurity incidents"
        ]
    },
    "q002": {
        "reference_answer": "Apple highlights supply chain risks including dependence on third-party manufacturing and logistics, component availability constraints, and disruptions from geopolitical or macroeconomic conditions.",
        "key_points": [
            "dependence on third-party suppliers or manufacturers",
            "component shortages or availability risks",
            "logistics or supply chain disruption",
            "geopolitical or macroeconomic impacts on supply chain"
        ],
        "unacceptable_claims": []
    },
    "q003": {
        "reference_answer": "Apple says adverse global and regional economic conditions can materially affect its business, operations, financial condition, and stock price.",
        "key_points": [
            "global and regional economic conditions matter",
            "adverse conditions may materially affect business",
            "financial condition may be harmed",
            "stock price may be affected"
        ],
        "unacceptable_claims": []
    },
    "q004": {
        "reference_answer": "Apple discusses changes in financial reporting standards including updates to segment expense disclosure requirements and accounting rules for internal-use software capitalization.",
        "key_points": [
            "segment expense disclosure changes",
            "internal-use software capitalization changes",
            "new accounting standards are discussed"
        ],
        "unacceptable_claims": []
    },
    "q005": {
        "reference_answer": "Apple says adverse economic conditions can materially adversely affect its business, results of operations, financial condition, and stock price.",
        "key_points": [
            "adverse economic conditions can hurt business",
            "results of operations may be affected",
            "financial condition may be affected",
            "stock price may be affected"
        ],
        "unacceptable_claims": []
    },
    "q006": {
        "reference_answer": "The filing does not provide enough evidence to determine Apple's exact dividend yield as of last quarter.",
        "key_points": [
            "insufficient evidence",
            "exact dividend yield cannot be determined from the filing"
        ],
        "unacceptable_claims": [
            "Apple's exact dividend yield is 2.4%",
            "the filing explicitly states the exact dividend yield"
        ]
    }
}

RETRIEVAL_LABELS = {
    q["id"]: {
        "chunk_ids": [],
        "parent_document_ids": [],
        "sections": [q["expected_section"]] if q.get("expected_section") else [],
    }
    for q in QUESTIONS
}


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    q_path = EVAL_DIR / "questions.json"
    g_path = EVAL_DIR / "gold_answers.json"
    l_path = EVAL_DIR / "retrieval_labels.json"

    q_path.write_text(json.dumps(QUESTIONS, indent=2, ensure_ascii=False), encoding="utf-8")
    g_path.write_text(json.dumps(GOLD_ANSWERS, indent=2, ensure_ascii=False), encoding="utf-8")
    l_path.write_text(json.dumps(RETRIEVAL_LABELS, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %d questions to %s", len(QUESTIONS), q_path)
    logger.info("Gold answers written to %s", g_path)
    logger.info("Retrieval labels written to %s", l_path)

    print(
        f"\n✓ Eval set seeded with {len(QUESTIONS)} questions.\n"
        "  Next steps:\n"
        "  1. Optionally add chunk_ids in retrieval_labels.json for stricter retrieval hit-rate\n"
        "  2. Run: python scripts/run_eval.py\n"
    )


if __name__ == "__main__":
    main()