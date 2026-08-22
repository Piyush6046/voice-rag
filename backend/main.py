"""
FastAPI entrypoint.

Run with:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    POST /api/query/voice   - multipart audio upload -> full voice RAG pipeline
    POST /api/query/text    - {"text": "..."} -> skips STT, useful for benchmarking
    GET  /api/health        - readiness check (index loaded, providers configured)
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import settings
from backend.pipeline.vectorstore import vector_store
from backend.pipeline.harness import harness
from backend.pipeline.schemas import PipelineResponse
from backend.pipeline.generation import _get_gemini_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    # 1. Connect to Pinecone (fast — no model download)
    try:
        vector_store.load()
        logger.info("Connected to Pinecone index '%s'.", settings.PINECONE_INDEX_NAME)
    except Exception as e:
        logger.warning("Could not connect to Pinecone: %s (server will still start)", e)

    # 2. Eagerly load embedding model + warm-up encode (runs in thread so it
    #    doesn't block the event loop during startup)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, vector_store.warm_up)
    except Exception as e:
        logger.warning("Embedder warm-up failed: %s", e)

    # 3. Pre-init Gemini client (cheap — just creates HTTP session)
    if settings.LLM_PROVIDER == "gemini":
        try:
            _get_gemini_client()
            logger.info("Gemini client ready (model: %s).", settings.GEMINI_MODEL)
        except Exception as e:
            logger.warning("Gemini client init failed: %s", e)

    logger.info("=== All systems ready. First query will be fast! ===")
    yield


app = FastAPI(title="HHG Voice RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextQuery(BaseModel):
    text: str


@app.get("/api/health")
async def health():
    index_stats = {}
    if vector_store.is_loaded():
        try:
            stats = vector_store.index.describe_index_stats()
            index_stats = {"total_vectors": stats.total_vector_count}
        except Exception:
            pass
    return {
        "status": "ok",
        "index_loaded": vector_store.is_loaded(),
        "index_name": settings.PINECONE_INDEX_NAME,
        **index_stats,
        "stt_provider": settings.STT_PROVIDER,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@app.post("/api/query/voice", response_model=PipelineResponse)
async def query_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file uploaded.")
    result = await harness.run_voice_query(audio_bytes, audio.filename or "audio.wav")
    return result


@app.post("/api/query/text", response_model=PipelineResponse)
async def query_text(body: TextQuery):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Empty text query.")
    result = await harness.run_text_query(body.text)
    return result


# Serve the simple demo frontend at / (built as static files, no build step needed)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
