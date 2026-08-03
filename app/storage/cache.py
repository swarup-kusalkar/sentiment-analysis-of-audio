"""Redis caching for audio hash and diarization results.

Avoids re-processing identical audio files and re-running diarization on known content (Phase 8).
"""

import json
import logging

import redis.asyncio as redis

from app.config import settings
from app.schemas.internal import DiarizationSegment

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 7 * 86400
_redis_client: redis.Redis | None = None


async def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def get_diarization(audio_hash: str) -> list[DiarizationSegment] | None:
    try:
        client = await _get_redis()
        data = await client.get(f"diarization:{audio_hash}")
        if data:
            return [DiarizationSegment(**item) for item in json.loads(data)]
    except Exception as exc:
        logger.warning("Redis get_diarization failed: %s", exc)
    return None


async def set_diarization(audio_hash: str, segments: list[DiarizationSegment]) -> None:
    try:
        client = await _get_redis()
        serializable = [seg.model_dump() for seg in segments]
        await client.set(
            f"diarization:{audio_hash}",
            json.dumps(serializable),
            ex=CACHE_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("Redis set_diarization failed: %s", exc)


async def get_analysis(audio_hash: str) -> dict | None:
    try:
        client = await _get_redis()
        data = await client.get(f"analysis:{audio_hash}")
        if data:
            return json.loads(data)
    except Exception as exc:
        logger.warning("Redis get_analysis failed: %s", exc)
    return None


async def set_analysis(audio_hash: str, analysis: dict) -> None:
    try:
        client = await _get_redis()
        await client.set(
            f"analysis:{audio_hash}",
            json.dumps(analysis),
            ex=CACHE_TTL_SECONDS,
        )
    except Exception as exc:
        logger.warning("Redis set_analysis failed: %s", exc)
