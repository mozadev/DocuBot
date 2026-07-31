"""
Guardrails for the RAG pipeline.

Two checkpoints, because input risk and output risk are different problems:

  check_input(question)   - runs before we spend a single token. Blocks prompt
                            injection and oversized input.
  check_output(answer, sources)
                          - runs before the answer reaches the user. Catches the
                            failure mode that actually matters in RAG: an answer
                            that sounds confident but is not supported by any
                            retrieved chunk.

Deliberately rule-based rather than an LLM judge. A judge would catch more, but
it doubles latency and cost on every turn and can itself be talked out of a
verdict. Rules are cheap, deterministic and testable; the LLM-judge upgrade is
noted in the README as future work.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.logger import logger

# Attempts to override the system prompt or exfiltrate it. These are matched
# against the user's question only, never against document content -- documents
# are data, and a PDF that happens to contain the words "ignore previous
# instructions" is not an attack on us.
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", "instruction override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|your)\s+", "instruction override"),
    (r"(reveal|show|print|repeat|output)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)", "prompt exfiltration"),
    (r"you\s+are\s+now\s+(a|an)\s+", "role reassignment"),
    (r"forget\s+(everything|all)\s+(you|above)", "instruction override"),
    (r"</?(system|assistant)>", "role tag injection"),
]

# Patterns redacted from the answer. Documents legitimately contain contact
# details, but echoing a credential back into a chat transcript is never useful.
PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[email redacted]"),
    (r"\b(?:\d[ -]*?){13,16}\b", "[card number redacted]"),
    (r"\b(sk|pk)-[A-Za-z0-9]{20,}\b", "[api key redacted]"),
]

MAX_QUESTION_CHARS = 2000

# Below this retrieval score the context is too weak to treat the answer as
# grounded. Tuned against text-embedding-3-small cosine scores, where a genuine
# topical match typically lands above ~0.3 and noise below it.
MIN_GROUNDING_SCORE = 0.25


@dataclass
class GuardrailResult:
    """Outcome of a guardrail checkpoint."""

    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "warnings": self.warnings,
        }


class RagGuardrails:
    """Input and output validation for document question-answering."""

    def __init__(
        self,
        min_grounding_score: float = MIN_GROUNDING_SCORE,
        max_question_chars: int = MAX_QUESTION_CHARS,
        redact_pii: bool = True,
    ) -> None:
        self._min_score = min_grounding_score
        self._max_chars = max_question_chars
        self._redact_pii = redact_pii

    def check_input(self, question: str) -> GuardrailResult:
        """Validate a user question before it reaches the LLM."""
        violations: list[str] = []

        if not question or not question.strip():
            return GuardrailResult(passed=False, violations=["Question is empty."], content="")

        if len(question) > self._max_chars:
            violations.append(
                f"Question exceeds {self._max_chars} characters ({len(question)}). "
                "Split it into smaller questions."
            )

        for pattern, label in INJECTION_PATTERNS:
            if re.search(pattern, question, re.IGNORECASE):
                violations.append(f"Possible prompt injection detected ({label}).")
                logger.warning("Guardrail blocked input: %s", label)
                break

        return GuardrailResult(
            passed=not violations,
            violations=violations,
            content=question.strip(),
        )

    def check_output(self, answer: str, sources: Sequence[Any]) -> GuardrailResult:
        """
        Validate a generated answer before returning it.

        The important check is grounding: if retrieval returned nothing useful
        but the model still produced a confident-looking answer, that answer is
        coming from parametric memory rather than the user's documents. In a
        "chat with your docs" product that is a wrong answer even when the fact
        it states happens to be true.
        """
        violations: list[str] = []
        warnings: list[str] = []
        content = answer or ""

        if not content.strip():
            return GuardrailResult(
                passed=False, violations=["Model returned an empty answer."], content=""
            )

        best_score = max((getattr(s, "score", 0.0) for s in sources), default=0.0)

        if not sources:
            if not self._is_refusal(content):
                violations.append(
                    "Answer is not grounded: no documents were retrieved, but the "
                    "model answered anyway."
                )
        elif best_score < self._min_score:
            warnings.append(
                f"Weak grounding: best retrieval score {best_score:.3f} is below "
                f"{self._min_score}. Treat this answer as low confidence."
            )

        if self._redact_pii:
            content, redacted = self._redact(content)
            if redacted:
                warnings.append(f"Redacted {redacted} sensitive value(s) from the answer.")

        return GuardrailResult(
            passed=not violations,
            violations=violations,
            warnings=warnings,
            content=content,
        )

    @staticmethod
    def _is_refusal(answer: str) -> bool:
        """True if the model correctly said it could not find the information."""
        markers = [
            "don't have", "do not have", "couldn't find", "could not find",
            "no information", "not found", "isn't in", "is not in",
            "no relevant", "unable to find", "not covered", "cannot answer",
        ]
        lowered = answer.lower()
        return any(m in lowered for m in markers)

    @staticmethod
    def _redact(text: str) -> tuple[str, int]:
        count = 0
        for pattern, replacement in PII_PATTERNS:
            text, n = re.subn(pattern, replacement, text)
            count += n
        return text, count
