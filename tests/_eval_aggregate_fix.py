# Helper — run as: python3 _eval_aggregate_fix.py
import re, sys

with open('tests/test_evaluation.py') as f:
    c = f.read()

# Fix RetrievalEvaluator aggregate test — positional args
c = c.replace(
    '        results = [\n            self.evaluator.evaluate(\"q1\", self._chunks([\"c001\"]), expected_chunk_ids=[\"c001\"]),\n            self.evaluator.evaluate(\"q2\", self._chunks([\"c099\"]), expected_chunk_ids=[\"c001\"]),\n        ]',
    '        results = [\n            self.evaluator.evaluate(question_id=\"q1\", retrieved_chunks=self._chunks([\"c001\"]), expected_chunk_ids=[\"c001\"]),\n            self.evaluator.evaluate(question_id=\"q2\", retrieved_chunks=self._chunks([\"c099\"]), expected_chunk_ids=[\"c001\"]),\n        ]'
)
# Fix AnswerEvaluator aggregate test
c = c.replace(
    '        results = [\n            self.evaluator.evaluate(\"q1\", answer=\"good answer\", reference_answer=\"good answer\"),\n            self.evaluator.evaluate(\"q2\", answer=\"bad\", reference_answer=\"completely different\"),\n        ]',
    '        results = [\n            self.evaluator.evaluate(question_id=\"q1\", answer=\"good answer\", reference_answer=\"good answer\"),\n            self.evaluator.evaluate(question_id=\"q2\", answer=\"bad\", reference_answer=\"completely different\"),\n        ]'
)
# Fix FaithfulnessEvaluator aggregate test
c = c.replace(
    '        results = [\n            self.evaluator.evaluate(\"q1\", answer=ctx, retrieved_chunks=self._chunks(ctx)),\n            self.evaluator.evaluate(\"q2\", answer=\"flying unicorns invented\",\n                                    retrieved_chunks=self._chunks(ctx)),\n        ]',
    '        results = [\n            self.evaluator.evaluate(question_id=\"q1\", answer=ctx, retrieved_chunks=self._chunks(ctx)),\n            self.evaluator.evaluate(question_id=\"q2\", answer=\"flying unicorns invented\",\n                                    retrieved_chunks=self._chunks(ctx)),\n        ]'
)
with open('tests/test_evaluation.py', 'w') as f:
    f.write(c)
print('Fixed evaluator keyword args')
