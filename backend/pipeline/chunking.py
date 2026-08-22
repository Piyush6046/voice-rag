"""
Chunking strategies for the MSMARCO-XI passage corpus.

We deliberately implement FOUR different chunkers instead of one naive
fixed-size splitter, because passages in this dataset vary a lot in length
and structure (short factoid passages vs. long explanatory ones):

1. FixedSizeChunker      - fixed token window + overlap. Cheap baseline.
2. SentenceWindowChunker - groups N sentences with sentence-level overlap,
                            preserves semantic boundaries better than raw
                            token windows.
3. SemanticChunker        - embeds sentences, merges consecutive sentences
                            while cosine similarity stays high, splits when
                            topic drifts. Good for long, multi-topic passages.
4. MetadataAwareChunker   - wraps any of the above but injects passage-level
                            metadata (source doc id, position, language) into
                            each chunk so retrieval/guardrails can use it.

`build_chunks_multi_strategy` runs all strategies over the corpus and tags
each resulting chunk with which strategy produced it, so we can compare
retrieval quality/latency per strategy at index-build time and pick the best
one(s) for the live index (see scripts/build_index.py).
"""
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter (avoids heavy nltk punkt download at import time).
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


@dataclass
class Chunk:
    chunk_id: str
    text: str
    strategy: str
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class FixedSizeChunker:
    """Naive-but-necessary baseline: fixed word window with overlap."""

    name = "fixed_size"

    def __init__(self, window_words: int = 60, overlap_words: int = 15):
        self.window_words = window_words
        self.overlap_words = overlap_words

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        words = text.split()
        if not words:
            return []
        step = max(1, self.window_words - self.overlap_words)
        chunks = []
        for start in range(0, len(words), step):
            piece = words[start:start + self.window_words]
            if not piece:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::{self.name}::{start}",
                text=" ".join(piece),
                strategy=self.name,
                doc_id=doc_id,
                metadata={"start_word": start, "end_word": start + len(piece)},
            ))
            if start + self.window_words >= len(words):
                break
        return chunks


class SentenceWindowChunker:
    """Groups whole sentences (2-4 at a time) with 1-sentence overlap.

    Keeps sentence boundaries intact, which fixed-size word windows don't -
    important for QA-style retrieval where cutting a sentence in half loses
    the exact fact being asked about.
    """

    name = "sentence_window"

    def __init__(self, sentences_per_chunk: int = 3, overlap_sentences: int = 1):
        self.n = sentences_per_chunk
        self.overlap = overlap_sentences

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        sents = _split_sentences(text)
        if not sents:
            return []
        step = max(1, self.n - self.overlap)
        chunks = []
        for i in range(0, len(sents), step):
            window = sents[i:i + self.n]
            if not window:
                continue
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::{self.name}::{i}",
                text=" ".join(window),
                strategy=self.name,
                doc_id=doc_id,
                metadata={"start_sentence": i, "n_sentences": len(window)},
            ))
            if i + self.n >= len(sents):
                break
        return chunks


class SemanticChunker:
    """Embedding-similarity based chunker.

    Walks sentence-by-sentence; keeps extending the current chunk while the
    next sentence's embedding stays cosine-similar to the running chunk
    centroid (topic hasn't drifted), otherwise starts a new chunk. This is
    the strategy that best handles long passages that cover more than one
    sub-topic (common in MS MARCO passages summarizing multi-part answers).
    """

    name = "semantic"

    def __init__(self, embedder, similarity_threshold: float = 0.55, max_sentences: int = 6):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.max_sentences = max_sentences

    def chunk(self, doc_id: str, text: str) -> List[Chunk]:
        sents = _split_sentences(text)
        if not sents:
            return []
        if len(sents) == 1:
            return [Chunk(
                chunk_id=f"{doc_id}::{self.name}::0",
                text=sents[0],
                strategy=self.name,
                doc_id=doc_id,
                metadata={"n_sentences": 1},
            )]

        embs = self.embedder.encode(sents, normalize_embeddings=True)
        chunks: List[Chunk] = []
        current = [0]
        centroid = embs[0]
        for idx in range(1, len(sents)):
            sim = float(np.dot(centroid, embs[idx]))
            if sim >= self.threshold and len(current) < self.max_sentences:
                current.append(idx)
                centroid = embs[current].mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            else:
                chunk_text = " ".join(sents[i] for i in current)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}::{self.name}::{current[0]}",
                    text=chunk_text,
                    strategy=self.name,
                    doc_id=doc_id,
                    metadata={"n_sentences": len(current)},
                ))
                current = [idx]
                centroid = embs[idx]
        if current:
            chunk_text = " ".join(sents[i] for i in current)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::{self.name}::{current[0]}",
                text=chunk_text,
                strategy=self.name,
                doc_id=doc_id,
                metadata={"n_sentences": len(current)},
            ))
        return chunks


class MetadataAwareChunker:
    """Wraps a base chunker, enriching every produced chunk with document-level
    metadata (source id, language, passage position, char length) so that
    guardrails and retrieval filters can use structured metadata instead of
    re-parsing text at query time.
    """

    def __init__(self, base_chunker, doc_metadata_fn=None):
        self.base = base_chunker
        self.doc_metadata_fn = doc_metadata_fn or (lambda doc_id, text: {})
        self.name = f"metadata_aware::{base_chunker.name}"

    def chunk(self, doc_id: str, text: str, extra_meta: Dict[str, Any] = None) -> List[Chunk]:
        base_chunks = self.base.chunk(doc_id, text)
        doc_meta = self.doc_metadata_fn(doc_id, text)
        for c in base_chunks:
            c.metadata.update(doc_meta)
            c.metadata["char_len"] = len(c.text)
            c.metadata["word_len"] = len(c.text.split())
            if extra_meta:
                c.metadata.update(extra_meta)
            c.strategy = self.name
        return base_chunks


def build_chunks_multi_strategy(
    doc_id: str,
    text: str,
    embedder,
    strategies: List[str] = None,
    extra_meta: Dict[str, Any] = None,
) -> List[Chunk]:
    """Run one or more chunking strategies over a single document and return
    the union of chunks, each tagged with `strategy` so retrieval can report
    which strategy the winning chunk came from.
    """
    strategies = strategies or ["fixed_size", "sentence_window", "semantic"]
    all_chunks: List[Chunk] = []

    if "fixed_size" in strategies:
        fc = FixedSizeChunker()
        mfc = MetadataAwareChunker(fc)
        all_chunks.extend(mfc.chunk(doc_id, text, extra_meta))

    if "sentence_window" in strategies:
        sw = SentenceWindowChunker()
        msw = MetadataAwareChunker(sw)
        all_chunks.extend(msw.chunk(doc_id, text, extra_meta))

    if "semantic" in strategies:
        sc = SemanticChunker(embedder)
        msc = MetadataAwareChunker(sc)
        all_chunks.extend(msc.chunk(doc_id, text, extra_meta))

    return all_chunks
