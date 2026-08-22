"""
Structured input/output contracts for every stage of the pipeline.
The harness passes these typed objects between stages instead of raw dicts,
so every stage's input/output is validated (fail fast, not silently pass bad data on).
"""
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class TranscriptionResult(BaseModel):
    text: str
    provider: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    latency_ms: float


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    strategy: str
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    chunks: List[RetrievedChunk]
    latency_ms: float


class GuardrailVerdict(BaseModel):
    passed: bool
    stage: Literal["input_safety", "off_topic", "grounding", "hallucination"]
    reason: str = ""
    score: Optional[float] = None


class GenerationResult(BaseModel):
    answer: str
    provider: str
    grounded: bool
    latency_ms: float


class StageTimings(BaseModel):
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    guardrail_ms: float = 0.0
    total_ms: float = 0.0


class PipelineResponse(BaseModel):
    status: Literal["answered", "refused", "error"]
    transcript: Optional[str] = None
    answer: Optional[str] = None
    refusal_reason: Optional[str] = None
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    guardrail_trace: List[GuardrailVerdict] = Field(default_factory=list)
    timings: StageTimings = Field(default_factory=StageTimings)
    retries: dict = Field(default_factory=dict)
    error: Optional[str] = None
