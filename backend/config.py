"""
Central configuration for the voice-RAG pipeline.
All secrets/config are read from environment variables (.env file supported).
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ---- Speech-to-text provider: "sarvam" or "elevenlabs" ----
    STT_PROVIDER: str = Field(default="sarvam")
    SARVAM_API_KEY: str = Field(default="")
    ELEVENLABS_API_KEY: str = Field(default="")

    # ---- Answer generation provider: "groq" | "gemini" | "anthropic" ----
    LLM_PROVIDER: str = Field(default="gemini")
    ANTHROPIC_API_KEY: str = Field(default="")
    GEMINI_API_KEY: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")
    ANTHROPIC_MODEL: str = Field(default="claude-3-5-haiku-latest")
    GEMINI_MODEL: str = Field(default="gemini-3.6-flash")
    GROQ_MODEL: str = Field(default="groq/compound-mini")

    # ---- Embedding model (local, no API needed => fast + free) ----
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # ---- Retrieval ----
    TOP_K: int = Field(default=5)
    INDEX_DIR: str = Field(default=os.path.join(os.path.dirname(__file__), "..", "data", "index"))

    # ---- Guardrails ----
    OFF_TOPIC_SIM_THRESHOLD: float = Field(default=0.28)  # below this => likely off-topic
    GROUNDING_OVERLAP_THRESHOLD: float = Field(default=0.15)  # min token overlap answer<->context

    # ---- Pinecone ----
    PINECONE_API_KEY: str = Field(default="")
    PINECONE_INDEX_NAME: str = Field(default="voice-rag")

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        extra = "ignore"


settings = Settings()
