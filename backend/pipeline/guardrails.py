"""
Guardrails run at two points in the pipeline:

  BEFORE retrieval (input-side):
    1. input_safety - blocks unsafe/inappropriate queries (jailbreak attempts,
       requests for harmful content, PII harvesting, etc.) using a keyword +
       pattern screen. This is intentionally conservative and fast (no LLM
       call) so it doesn't blow the latency budget.
    2. off_topic - if the *best* retrieval score for the query is below a
       similarity threshold, the corpus almost certainly doesn't contain the
       answer, so we refuse rather than let the LLM hallucinate one.

  AFTER generation (output-side):
    3. grounding - checks that the generated answer actually shares
       vocabulary/entities with the retrieved context (token-overlap +
       n-gram containment heuristic). Cheap and dependency-free.
    4. hallucination - a stricter LLM-based self-check ("does this answer
       follow ONLY from the given context? yes/no") used as a second line of
       defense when the cheap grounding heuristic is inconclusive.

The system is designed to say "I don't have enough information to answer
that from the provided dataset" rather than guess - that refusal path is
itself a first-class, testable output of the pipeline.
"""
from __future__ import annotations
import re
from typing import List, Tuple

from backend.pipeline.schemas import RetrievedChunk, GuardrailVerdict
from backend.config import settings

# Deliberately conservative pattern list - catches common jailbreak / unsafe
# request phrasing without needing a network call. This is a first line of
# defense, not a full safety classifier.
_UNSAFE_PATTERNS = [
    r"\bignore\s+(?:all|any|the|your|previous|prior|above|given)?\s*(?:all|any|the|your|previous|prior|above|given)?\s*instructions\b",
    r"\bact as (an? )?(unfiltered|jailbroken|dan)\b",
    r"\bhow (do|to) (i |you )?(make|build|synthesi[sz]e).{0,30}(bomb|explosive|weapon|virus|malware)\b",
    r"\b(kill|harm|hurt) (myself|yourself|someone)\b",
    r"\bcredit card number\b",
    r"\bsocial security number\b",
    r"\bsteal (a |someone'?s )?(identity|password|credentials)\b",
]
_UNSAFE_RE = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)


def check_input_safety(query: str) -> GuardrailVerdict:
    if _UNSAFE_RE.search(query):
        return GuardrailVerdict(
            passed=False,
            stage="input_safety",
            reason="Query matched an unsafe/jailbreak pattern and was blocked before retrieval.",
        )
    return GuardrailVerdict(passed=True, stage="input_safety", reason="No unsafe pattern matched.")


def check_off_topic(retrieved: List[RetrievedChunk]) -> GuardrailVerdict:
    if not retrieved:
        return GuardrailVerdict(
            passed=False, stage="off_topic", reason="No chunks retrieved at all.", score=0.0
        )
    best_score = max(c.score for c in retrieved)
    if best_score < settings.OFF_TOPIC_SIM_THRESHOLD:
        return GuardrailVerdict(
            passed=False,
            stage="off_topic",
            reason=(
                f"Best retrieval similarity {best_score:.3f} is below threshold "
                f"{settings.OFF_TOPIC_SIM_THRESHOLD} - the corpus likely doesn't cover this query."
            ),
            score=best_score,
        )
    return GuardrailVerdict(passed=True, stage="off_topic", reason="Retrieved context is relevant enough.", score=best_score)


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def check_grounding(answer: str, retrieved: List[RetrievedChunk]) -> GuardrailVerdict:
    if not answer.strip():
        return GuardrailVerdict(passed=False, stage="grounding", reason="Empty answer.", score=0.0)

    context_tokens = set()
    for c in retrieved:
        context_tokens |= _tokenize(c.text)
    answer_tokens = _tokenize(answer)
    # Ignore ultra-common stopword-ish tokens from the denominator so grounding
    # score isn't inflated by "the/is/a/of" overlap alone.
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and",
            "for", "on", "with", "as", "by", "it", "this", "that", "be", "or"}
    meaningful = answer_tokens - stop
    if not meaningful:
        return GuardrailVerdict(passed=True, stage="grounding", reason="Answer too short to evaluate; passing through.", score=1.0)

    overlap = len(meaningful & context_tokens) / len(meaningful)
    if overlap < settings.GROUNDING_OVERLAP_THRESHOLD:
        return GuardrailVerdict(
            passed=False,
            stage="grounding",
            reason=f"Only {overlap:.0%} of the answer's content words appear in retrieved context - likely ungrounded.",
            score=overlap,
        )
    return GuardrailVerdict(passed=True, stage="grounding", reason=f"{overlap:.0%} token overlap with retrieved context.", score=overlap)


async def check_hallucination_llm(question: str, answer: str, retrieved: List[RetrievedChunk], llm_client) -> GuardrailVerdict:
    """Second-line, LLM-based self-check. Only invoked when the cheap grounding
    heuristic is borderline, to avoid spending an extra LLM call (and latency
    budget) on every single query.
    """
    context = "\n\n".join(c.text for c in retrieved)
    prompt = (
        "You are a strict fact-checker. Given CONTEXT and an ANSWER to a QUESTION, "
        "reply with exactly one word: YES if the answer is fully supported by the "
        "context, or NO if the answer contains any claim not present in the context.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSupported? (YES/NO):"
    )
    try:
        verdict_text = await llm_client.raw_complete(prompt, max_tokens=5)
        supported = verdict_text.strip().upper().startswith("Y")
        return GuardrailVerdict(
            passed=supported,
            stage="hallucination",
            reason="LLM self-check: " + ("supported by context." if supported else "NOT supported by context."),
        )
    except Exception as e:
        # Fail open on the *check* itself (don't crash the request), but log
        # that the check didn't run - callers should treat this as "unknown".
        return GuardrailVerdict(passed=True, stage="hallucination", reason=f"Hallucination check skipped (error: {e})")


def run_input_guardrails(query: str) -> Tuple[bool, List[GuardrailVerdict]]:
    trace = [check_input_safety(query)]
    passed = trace[-1].passed
    return passed, trace


def run_retrieval_guardrails(retrieved: List[RetrievedChunk]) -> Tuple[bool, List[GuardrailVerdict]]:
    trace = [check_off_topic(retrieved)]
    passed = trace[-1].passed
    return passed, trace
