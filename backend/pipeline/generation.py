"""
Answer generation stage. Provider-agnostic: switch between Anthropic (Claude)
and Google (Gemini) with LLM_PROVIDER in .env. The prompt hard-constrains the
model to only use retrieved context and to explicitly say when it can't
answer, so the guardrail layer has less work to do and the model is less
likely to hallucinate in the first place.
"""
from __future__ import annotations
import time
import logging
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.pipeline.schemas import RetrievedChunk, GenerationResult

logger = logging.getLogger("generation")

_SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant for a voice RAG system. "
    "Answer the user's question using ONLY the provided context passages. "
    "If the context does not contain enough information to answer confidently, "
    "reply exactly with: \"I don't have enough information to answer this.\" "
    "Do not use outside knowledge. Give a complete, well-formed answer. "
    "Do not cut off mid-sentence. Aim for 2-5 sentences depending on complexity."
)

_SYSTEM_PROMPT_FALLBACK = (
    "You are a helpful conversational assistant. Answer the user's question directly using your own general knowledge.\n"
    "If the question asks about specific, private, or domain-specific facts that you do not know, "
    "or if you do not have enough information to answer confidently, reply exactly with: "
    "\"I don't have enough information to answer this.\"\n"
    "Do not make up facts. Be brief and conversational (1-3 sentences)."
)

# Max chars per chunk to include in the prompt — keeps prompts short and fast.
_MAX_CHUNK_CHARS = 350


def _build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    # Truncate each chunk so the prompt stays small (faster, cheaper)
    context = "\n\n".join(
        f"[{i+1}] {c.text[:_MAX_CHUNK_CHARS]}{'...' if len(c.text) > _MAX_CHUNK_CHARS else ''}"
        for i, c in enumerate(chunks)
    )
    return (
        f"CONTEXT PASSAGES:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above:"
        f"if answer is not in context, return: 'I don't have enough information in the provided dataset to answer that.'"
    )


# --------------------------------------------------------------------------
# Singleton Gemini client — created once at import time for fast reuse.
# --------------------------------------------------------------------------
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logger.info("Gemini client initialised (model: %s)", settings.GEMINI_MODEL)
    return _gemini_client


class LLMClient:
    """Thin wrapper so the guardrail hallucination-checker and the main
    generation stage can share one client (and one retry policy)."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    async def raw_complete(self, prompt: str, max_tokens: int = 500, system: str = None) -> str:
        if self.provider == "groq":
            return await self._groq_complete(prompt, max_tokens, system)
        elif self.provider == "gemini":
            return await self._gemini_complete(prompt, max_tokens, system)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    # async def _anthropic_complete(self, prompt: str, max_tokens: int, system: str) -> str:
    #     import anthropic
    #     if not settings.ANTHROPIC_API_KEY:
    #         raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to backend/.env")
    #     client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    #     resp = await client.messages.create(
    #         model=settings.ANTHROPIC_MODEL,
    #         max_tokens=max_tokens,
    #         system=system or _SYSTEM_PROMPT,
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     return "".join(block.text for block in resp.content if block.type == "text").strip()

    async def _groq_complete(self, prompt: str, max_tokens: int, system: str) -> str:
        """Groq inference — typically 300-600ms, much faster than Gemini."""
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Add it to backend/.env")
        from groq import AsyncGroq
        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        resp = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system or _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    async def _gemini_complete(self, prompt: str, max_tokens: int, system: str) -> str:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set. Add it to backend/.env")
        from google.genai import types
        client = _get_gemini_client()
        full_prompt = (system or _SYSTEM_PROMPT) + "\n\n" + prompt
        resp = await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.2,
            ),
        )
        return (resp.text or "").strip()


llm_client = LLMClient()


async def generate_answer(question: str, chunks: List[RetrievedChunk]) -> GenerationResult:
    t0 = time.perf_counter()
    prompt = _build_prompt(question, chunks)
    answer = await llm_client.raw_complete(prompt, max_tokens=500)
    latency_ms = (time.perf_counter() - t0) * 1000
    refused = answer.strip().lower().startswith("i don't have enough information")
    return GenerationResult(
        answer=answer,
        provider=llm_client.provider,
        grounded=not refused,
        latency_ms=latency_ms,
    )


async def generate_fallback_answer(question: str) -> GenerationResult:
    t0 = time.perf_counter()
    answer = await llm_client.raw_complete(question, max_tokens=500, system=_SYSTEM_PROMPT_FALLBACK)
    latency_ms = (time.perf_counter() - t0) * 1000
    refused = answer.strip().lower().startswith("i don't have enough information")
    return GenerationResult(
        answer=answer,
        provider=llm_client.provider,
        grounded=not refused,
        latency_ms=latency_ms,
    )
