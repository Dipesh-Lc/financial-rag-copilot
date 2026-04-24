#!/usr/bin/env python
"""
run_eval.py
Run the evaluation suite and print a formatted summary.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --strategy multi_query
    python scripts/run_eval.py --questions data/eval/questions.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vectorstore.chroma_store import get_vector_store
from app.vectorstore.retriever import FilingRetriever
from app.services.query_service import QueryService
from app.evaluation.eval_runner import EvalRunner
from app.logging_config import get_logger

logger = get_logger("run_eval")

DIVIDER = "─" * 55


def print_summary(results: dict) -> None:
    summary = results.get("summary", {})
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS  |  strategy={results.get('strategy', '?')}")
    print(f"  Questions evaluated: {results.get('num_questions', 0)}")
    print("=" * 55)

    ret = summary.get("retrieval", {})
    print(f"\n{DIVIDER}")
    print("  RETRIEVAL")
    print(DIVIDER)
    print(f"  retrieval_hit_rate      : {ret.get('retrieval_hit_rate', 'n/a')}")
    print(f"  section_hit_rate        : {ret.get('section_hit_rate', 'n/a')}")
    print(f"  parent_hit_rate         : {ret.get('parent_hit_rate', 'n/a')}")
    print(f"  mean_precision_at_k     : {ret.get('mean_precision_at_k', 'n/a')}")
    print(f"  mean_recall_at_k        : {ret.get('mean_recall_at_k', 'n/a')}")

    ans = summary.get("answer", {})
    print(f"\n{DIVIDER}")
    print("  ANSWER QUALITY")
    print(DIVIDER)
    print(f"  mean_answer_relevance   : {ans.get('mean_answer_relevance', 'n/a')}")
    print(f"  mean_key_point_coverage : {ans.get('mean_key_point_coverage', 'n/a')}")
    print(f"  insufficient_context    : {ans.get('insufficient_context_compliance_rate', 'n/a')}")
    print(f"  hallucination_rate      : {ans.get('hallucination_rate', 'n/a')}")

    faith = summary.get("faithfulness", {})
    print(f"\n{DIVIDER}")
    print("  FAITHFULNESS")
    print(DIVIDER)
    print(f"  supported_sentence_ratio: {faith.get('mean_supported_sentence_ratio', 'n/a')}")
    print(f"  citation_presence_rate  : {faith.get('citation_presence_rate', 'n/a')}")
    print(f"  unsupported_answer_rate : {faith.get('unsupported_answer_rate', 'n/a')}")

    schema_v = summary.get("schema_validity_rate")
    print(f"\n{DIVIDER}")
    print("  SCHEMA")
    print(DIVIDER)
    print(f"  schema_validity_rate    : {schema_v}")
    print(f"\n{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation suite")
    parser.add_argument(
        "--strategy",
        choices=["similarity", "multi_query", "parent_doc"],
        default="similarity",
    )
    parser.add_argument("--questions", type=Path, default=None)
    args = parser.parse_args()

    store = get_vector_store()
    retriever = FilingRetriever(store)
    query_service = QueryService(retriever)

    runner = EvalRunner(query_service=query_service, retriever=retriever)
    results = runner.run(questions_path=args.questions, strategy=args.strategy)

    if results:
        print_summary(results)
    else:
        print("No results — check logs and make sure data/eval/questions.json exists.")


if __name__ == "__main__":
    main()
