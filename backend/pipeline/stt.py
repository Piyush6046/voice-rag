"""
Speech-to-text stage. Supports Sarvam AI (good for Indic languages, pairs well
with the AI4Bharat MSMARCO-XI dataset) and ElevenLabs, switchable via config.
"""
import time
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.config import settings
from backend.pipeline.schemas import TranscriptionResult


class STTError(Exception):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.3, max=2),
    retry=retry_if_exception_type((httpx.HTTPError, STTError)),
    reraise=True,
)
async def _transcribe_sarvam(audio_bytes: bytes, filename: str) -> TranscriptionResult:
    if not settings.SARVAM_API_KEY:
        raise STTError("SARVAM_API_KEY not set. Add it to backend/.env")

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": settings.SARVAM_API_KEY}
    files = {"file": (filename, audio_bytes, "audio/wav")}
    data = {"model": "saarika:v2", "language_code": "unknown"}

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, headers=headers, files=files, data=data)
    latency_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        raise STTError(f"Sarvam STT failed: {resp.status_code} {resp.text[:200]}")

    body = resp.json()
    text = body.get("transcript", "").strip()
    if not text:
        raise STTError("Sarvam returned empty transcript")

    return TranscriptionResult(
        text=text,
        provider="sarvam",
        language=body.get("language_code"),
        latency_ms=latency_ms,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.3, min=0.3, max=2),
    retry=retry_if_exception_type((httpx.HTTPError, STTError)),
    reraise=True,
)
async def _transcribe_elevenlabs(audio_bytes: bytes, filename: str) -> TranscriptionResult:
    if not settings.ELEVENLABS_API_KEY:
        raise STTError("ELEVENLABS_API_KEY not set. Add it to backend/.env")

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": settings.ELEVENLABS_API_KEY}
    files = {"file": (filename, audio_bytes, "audio/wav")}
    data = {"model_id": "scribe_v1"}

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(url, headers=headers, files=files, data=data)
    latency_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        raise STTError(f"ElevenLabs STT failed: {resp.status_code} {resp.text[:200]}")

    body = resp.json()
    text = body.get("text", "").strip()
    if not text:
        raise STTError("ElevenLabs returned empty transcript")

    return TranscriptionResult(
        text=text,
        provider="elevenlabs",
        language=body.get("language_code"),
        latency_ms=latency_ms,
    )


async def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> TranscriptionResult:
    if settings.STT_PROVIDER == "sarvam":
        return await _transcribe_sarvam(audio_bytes, filename)
    elif settings.STT_PROVIDER == "elevenlabs":
        return await _transcribe_elevenlabs(audio_bytes, filename)
    else:
        raise STTError(f"Unknown STT_PROVIDER: {settings.STT_PROVIDER}")
