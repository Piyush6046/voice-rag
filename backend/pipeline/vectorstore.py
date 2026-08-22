"""
Pinecone-backed vector store.
The embedder is preloaded at server startup (warm_up()) so the first query
is just as fast as every subsequent query.
"""
from __future__ import annotations
import logging
import time
from functools import lru_cache
from typing import List

from pinecone import Pinecone
import numpy as np

from backend.config import settings
from backend.pipeline.schemas import RetrievedChunk

logger = logging.getLogger("vectorstore")


@lru_cache(maxsize=128)
def _cached_embed(text: str, model_name: str):
    """Cache query embeddings — same query text never re-encodes."""
    return None  # filled by VectorStore after model loads


class VectorStore:
    def __init__(self):
        self.pc: Pinecone | None = None
        self.index = None
        self._embedder = None
        self._embed_cache: dict = {}  # query text → embedding list

    @property
    def embedder(self):
        """Return the embedding model, loading it if not yet initialised."""
        if self._embedder is None:
            self._load_embedder()
        return self._embedder

    def _load_embedder(self):
        from fastembed import TextEmbedding
        logger.info("Loading embedding model: %s …", settings.EMBEDDING_MODEL)
        self._embedder = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
        logger.info("Embedding model ready.")

    def warm_up(self):
        """Eagerly load the embedding model and run a dummy encode so the
        first real query incurs zero model-loading overhead."""
        self._load_embedder()
        logger.info("Warming up embedder with dummy encode…")
        list(self._embedder.embed(["warm up"]))
        logger.info("Embedder warm-up complete.")

    def load(self):
        """Connect to Pinecone. Fast — no model download."""
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY must be set in backend/.env")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
        logger.info("Connected to Pinecone index: %s", settings.PINECONE_INDEX_NAME)

    def is_loaded(self) -> bool:
        return self.index is not None

    def search(self, query: str, top_k: int = None):
        top_k = top_k or settings.TOP_K
        t0 = time.perf_counter()

        # E5 models require a "query: " prefix for asymmetric search
        if "e5" in settings.EMBEDDING_MODEL.lower():
            query_input = f"query: {query}"
        else:
            query_input = query

        # Check cache first — avoids re-encoding repeated queries
        cache_key = query_input
        if cache_key in self._embed_cache:
            q_emb_list = self._embed_cache[cache_key]
            logger.debug("Embedding cache hit for query.")
        else:
            q_emb = list(self.embedder.embed([query_input]))
            q_emb_list = np.asarray(q_emb, dtype="float32")[0].tolist()
            self._embed_cache[cache_key] = q_emb_list
            # Keep cache bounded at 256 entries
            if len(self._embed_cache) > 256:
                self._embed_cache.pop(next(iter(self._embed_cache)))
        encode_ms = (time.perf_counter() - t0) * 1000

        # Search Pinecone
        t1 = time.perf_counter()
        response = self.index.query(vector=q_emb_list, top_k=top_k, include_metadata=True)
        pinecone_ms = (time.perf_counter() - t1) * 1000
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("Retrieval: encode=%.0fms  pinecone=%.0fms  total=%.0fms",
                    encode_ms, pinecone_ms, latency_ms)

        # Map to schema
        results: List[RetrievedChunk] = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(RetrievedChunk(
                chunk_id=match.id,
                text=meta.get("text", ""),
                score=float(match.score),
                strategy=meta.get("strategy", "precomputed"),
                metadata={k: v for k, v in meta.items() if k not in ("text", "strategy")}
            ))
        return results, latency_ms


# Process-wide singleton, populated at FastAPI startup.
vector_store = VectorStore()
