"""
test_evaluation.py
Tests for all three evaluator classes: RetrievalEvaluator, AnswerEvaluator, FaithfulnessEvaluator.
Pure logic — no network, no LLM.
"""

import pytest


# ── RetrievalEvaluator ────────────────────────────────────────────────────────

class TestRetrievalEvaluator:
    def setup_method(self):
        from app.evaluation.retrieval_eval import RetrievalEvaluator
        self.evaluator = RetrievalEvaluator()

    def _chunks(self, ids):
        return [{"chunk_id": cid, "section_name": "Risk Factors",
                 "parent_document_id": f"doc-{cid}"} for cid in ids]

    def test_perfect_hit(self):
        result = self.evaluator.evaluate(
            question_id="q1",
            retrieved_chunks=self._chunks(["c001", "c002"]),
            expected_chunk_ids=["c001"],
        )
        assert result.hit_at_k is True
        assert result.precision_at_k == 0.5
        assert result.recall_at_k == 1.0

    def test_zero_hit(self):
        result = self.evaluator.evaluate(
            question_id="q1",
            retrieved_chunks=self._chunks(["c099"]),
            expected_chunk_ids=["c001"],
        )
        assert result.hit_at_k is False
        assert result.recall_at_k == 0.0

    def test_no_expected_chunks_section_fallback(self):
        chunks = [{"chunk_id": "c1", "section_name": "Risk Factors", "parent_document_id": None}]
        result = self.evaluator.evaluate(
            question_id="q1",
            retrieved_chunks=chunks,
            expected_chunk_ids=[],
            expected_sections=["Risk Factors"],
        )
        assert result.section_hit is True

    def test_aggregate_hit_rate(self):
        from app.evaluation.retrieval_eval import RetrievalEvalResult
        results = [
            self.evaluator.evaluate(question_id="q1", retrieved_chunks=self._chunks(["c001"]), expected_chunk_ids=["c001"]),
            self.evaluator.evaluate(question_id="q2", retrieved_chunks=self._chunks(["c099"]), expected_chunk_ids=["c001"]),
        ]
        agg = self.evaluator.aggregate(results)
        assert agg["retrieval_hit_rate"] == 0.5
        assert agg["count"] == 2

    def test_aggregate_empty(self):
        agg = self.evaluator.aggregate([])
        assert agg["count"] == 0
        assert agg["retrieval_hit_rate"] == 0.0


# ── AnswerEvaluator ───────────────────────────────────────────────────────────

class TestAnswerEvaluator:
    def setup_method(self):
        from app.evaluation.answer_eval import AnswerEvaluator
        self.evaluator = AnswerEvaluator()

    def test_perfect_relevance(self):
        answer = "apple faces competition in smartphones"
        result = self.evaluator.evaluate(
            question_id="q1",
            answer=answer,
            reference_answer=answer,
        )
        assert result.answer_relevance == 1.0

    def test_zero_relevance(self):
        result = self.evaluator.evaluate(
            question_id="q1",
            answer="completely unrelated text here",
            reference_answer="apple faces competition smartphones market share",
        )
        assert result.answer_relevance < 0.3

    def test_key_point_coverage(self):
        result = self.evaluator.evaluate(
            question_id="q1",
            answer="Apple faces cyber risk and supply chain disruption",
            key_points=["cyber risk", "supply chain"],
        )
        assert result.key_point_coverage == 1.0
        assert len(result.matched_key_points) == 2

    def test_insufficient_context_compliance(self):
        result = self.evaluator.evaluate(
            question_id="q1",
            answer="Insufficient evidence to determine this from the filings.",
            expects_insufficient_context=True,
        )
        assert result.insufficient_context_compliance is True

    def test_hallucination_flag(self):
        result = self.evaluator.evaluate(
            question_id="q1",
            answer="Apple reported revenue of 999 billion dollars",
            unacceptable_claims=["999 billion"],
        )
        assert result.hallucination_flag is True

    def test_aggregate(self):
        results = [
            self.evaluator.evaluate(question_id="q1", answer="good answer", reference_answer="good answer"),
            self.evaluator.evaluate(question_id="q2", answer="bad", reference_answer="completely different"),
        ]
        agg = self.evaluator.aggregate(results)
        assert agg["count"] == 2
        assert 0.0 <= agg["mean_answer_relevance"] <= 1.0


# ── FaithfulnessEvaluator ─────────────────────────────────────────────────────

class TestFaithfulnessEvaluator:
    def setup_method(self):
        from app.evaluation.faithfulness_eval import FaithfulnessEvaluator
        self.evaluator = FaithfulnessEvaluator()

    def _chunks(self, text):
        return [{"text": text, "chunk_id": "c1"}]

    def test_high_faithfulness(self):
        ctx = "apple faces competition smartphones android rivals market share devices"
        result = self.evaluator.evaluate(
            question_id="q1",
            answer=ctx,
            retrieved_chunks=self._chunks(ctx),
        )
        assert result.supported_sentence_ratio >= 0.8

    def test_unfaithful_answer(self):
        result = self.evaluator.evaluate(
            question_id="q1",
            answer="Flying elephants purple unicorns magic rainbows completely invented claims.",
            retrieved_chunks=self._chunks("apple revenue cloud azure quarterly earnings financial"),
        )
        assert result.supported_sentence_ratio < 0.5

    def test_empty_context_graceful(self):
        result = self.evaluator.evaluate(
            question_id="q1",
            answer="Some answer.",
            retrieved_chunks=[],
        )
        assert isinstance(result.supported_sentence_ratio, float)

    def test_aggregate(self):
        ctx = "apple faces risk factors regulatory cybersecurity market competition"
        results = [
            self.evaluator.evaluate(question_id="q1", answer=ctx, retrieved_chunks=self._chunks(ctx)),
            self.evaluator.evaluate(question_id="q2", answer="flying unicorns invented",
                                    retrieved_chunks=self._chunks(ctx)),
        ]
        agg = self.evaluator.aggregate(results)
        assert agg["count"] == 2
        assert 0.0 <= agg["mean_supported_sentence_ratio"] <= 1.0
