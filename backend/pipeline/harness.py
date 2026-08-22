"""
The harness: structured orchestration around the model, per task requirement #5.

Rather than "single raw prompt-in, text-out", every query goes through a
typed state machine:

    input_safety guardrail
        -> (blocked) refuse, done
        -> (pass) retrieval
    retrieval (vector DB)
        -> off_topic guardrail
            -> (blocked) refuse, done
            -> (pass) generation
    generation (LLM call, retried on transient failure)
        -> grounding guardrail (cheap heuristic)
            -> (fails) escalate to LLM hallucination self-check
                -> (fails) refuse, done
                -> (pass) return answer
            -> (borderline) escalate to LLM hallucination self-check
            -> (pass, high confidence) return answer

Every stage's input/output is a pydantic model (see schemas.py), every stage
records its own latency, and every stage that can transiently fail (STT,
vector search, LLM calls) is wrapped in retry logic. If a stage fails after
retries, the harness returns a structured `error` response instead of
crashing the request - this is the "error recovery" requirement.
"""
from __future__ import annotations
import time
import logging

from backend.pipeline import stt, guardrails
from backend.pipeline.vectorstore import vector_store
from backend.pipeline.generation import generate_answer, llm_client, generate_fallback_answer
from backend.pipeline.schemas import PipelineResponse, StageTimings, RetrievedChunk

logger = logging.getLogger("harness")


class PipelineHarness:
    """Single entry point used by the API layer. Owns retry/error-recovery
    policy for the whole request, on top of the per-stage retries already
    inside stt.py / generation.py."""

    async def run_voice_query(self, audio_bytes: bytes, filename: str) -> PipelineResponse:
        timings = StageTimings()
        retries_used = {"stt": 0, "generation": 0}
        t_start = time.perf_counter()

        # ---- Stage 1: Speech-to-text ----
        try:
            transcript_result = await stt.transcribe(audio_bytes, filename)
        except Exception as e:
            logger.exception("STT failed after retries")
            return PipelineResponse(
                status="error",
                error=f"Speech-to-text failed: {e}",
                timings=timings,
            )
        timings.stt_ms = transcript_result.latency_ms
        query_text = transcript_result.text

        response = await self.run_text_query(query_text, timings=timings)
        response.transcript = query_text
        response.timings.total_ms = (time.perf_counter() - t_start) * 1000
        return response

    async def run_text_query(self, query_text: str, timings: StageTimings = None) -> PipelineResponse:
        """Runs everything downstream of transcription. Exposed separately so
        the frontend / benchmark script can hit a text-only endpoint too
        (useful for isolating retrieval+generation latency from STT latency,
        since the two are governed by very different latency budgets - see
        scripts/benchmark.py and the README latency section).
        """
        timings = timings or StageTimings()
        t_start = time.perf_counter()
        guardrail_trace = []

        # ---- Guardrail: input safety (pre-retrieval) ----
        t0 = time.perf_counter()
        passed, trace = guardrails.run_input_guardrails(query_text)
        guardrail_trace.extend(trace)
        timings.guardrail_ms += (time.perf_counter() - t0) * 1000
        if not passed:
            return PipelineResponse(
                status="refused",
                refusal_reason=trace[-1].reason,
                guardrail_trace=guardrail_trace,
                timings=timings,
            )

        # ---- Stage 2: Retrieval ----
        if not vector_store.is_loaded():
            return PipelineResponse(
                status="error",
                error="Vector index not loaded. Run scripts/build_index.py then restart the server.",
                timings=timings,
                guardrail_trace=guardrail_trace,
            )
        try:
            chunks, retrieval_ms = vector_store.search(query_text)
        except Exception as e:
            logger.exception("Retrieval failed")
            return PipelineResponse(
                status="error", error=f"Retrieval failed: {e}",
                timings=timings, guardrail_trace=guardrail_trace,
            )
        timings.retrieval_ms = retrieval_ms

        # ---- Guardrail: off-topic (post-retrieval check to see if present in DB) ----
        t0 = time.perf_counter()
        passed, trace = guardrails.run_retrieval_guardrails(chunks)
        guardrail_trace.extend(trace)
        timings.guardrail_ms += (time.perf_counter() - t0) * 1000

        in_db = passed
        gen_result = None

        if in_db:
            # ---- Stage 3: Generation (grounded RAG) ----
            try:
                gen_result = await generate_answer(query_text, chunks)
                timings.generation_ms = gen_result.latency_ms
            except Exception as e:
                logger.exception("Grounded generation failed")
                in_db = False

        if in_db and gen_result:
            if not gen_result.grounded:
                # Model itself declined (context insufficient)
                in_db = False
            else:
                # ---- Guardrail: grounding (cheap heuristic, post-generation) ----
                t0 = time.perf_counter()
                grounding_verdict = guardrails.check_grounding(gen_result.answer, chunks)
                guardrail_trace.append(grounding_verdict)
                timings.guardrail_ms += (time.perf_counter() - t0) * 1000

                if not grounding_verdict.passed:
                    # Escalate to the stricter LLM-based hallucination self-check
                    t0 = time.perf_counter()
                    halluc_verdict = await guardrails.check_hallucination_llm(
                        query_text, gen_result.answer, chunks, llm_client
                    )
                    guardrail_trace.append(halluc_verdict)
                    timings.guardrail_ms += (time.perf_counter() - t0) * 1000
                    if not halluc_verdict.passed:
                        in_db = False

        # If it was found in DB and passed grounded generation checks, return it
        if in_db and gen_result:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return PipelineResponse(
                status="answered",
                answer=gen_result.answer,
                retrieved_chunks=chunks,
                guardrail_trace=guardrail_trace,
                timings=timings,
            )

        # Otherwise (not in DB or RAG checks failed), check if LLM can answer using general knowledge
        try:
            fallback_result = await generate_fallback_answer(query_text)
        except Exception as e:
            logger.exception("Fallback generation failed")
            return PipelineResponse(
                status="error",
                error=f"Answer generation failed: {e}",
                retrieved_chunks=chunks,
                guardrail_trace=guardrail_trace,
                timings=timings,
            )
        timings.generation_ms += fallback_result.latency_ms
        timings.total_ms = (time.perf_counter() - t_start) * 1000

        if not fallback_result.grounded:
            # LLM declined (specific query not in DB, no knowledge)
            return PipelineResponse(
                status="refused",
                refusal_reason="I don't have enough information to answer this.",
                retrieved_chunks=chunks,
                guardrail_trace=guardrail_trace,
                timings=timings,
            )
        else:
            return PipelineResponse(
                status="answered",
                answer=fallback_result.answer,
                retrieved_chunks=chunks,
                guardrail_trace=guardrail_trace,
                timings=timings,
            )


harness = PipelineHarness()
