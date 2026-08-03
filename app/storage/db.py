"""PostgreSQL connection and result persistence.

Saves the final JSON analysis results after the full pipeline completes (Phase 8).
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.config import settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: sessionmaker | None = None


async def init_db() -> None:
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
        )
        _session_factory = sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
        # Create table if not exists
        async with _engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id BIGSERIAL PRIMARY KEY,
                    audio_hash VARCHAR(64) UNIQUE NOT NULL,
                    original_name VARCHAR(255),
                    duration_seconds REAL,
                    speaker_count INT,
                    overall_confidence REAL,
                    needs_human_review BOOLEAN,
                    result_json JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_analyses_audio_hash ON analyses(audio_hash)
            """))
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC)
            """))
        logger.info("Database initialized")


@asynccontextmanager
async def get_session():
    if _session_factory is None:
        await init_db()
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def save_analysis(
    audio_hash: str,
    original_name: str,
    duration_seconds: float,
    speaker_count: int,
    overall_confidence: float,
    needs_human_review: bool,
    result_json: dict,
) -> Optional[int]:
    try:
        async with get_session() as session:
            result = await session.execute(
                text("""
                    INSERT INTO analyses (
                        audio_hash, original_name, duration_seconds,
                        speaker_count, overall_confidence, needs_human_review, result_json
                    ) VALUES (
                        :audio_hash, :original_name, :duration_seconds,
                        :speaker_count, :overall_confidence, :needs_human_review, :result_json
                    )
                    ON CONFLICT (audio_hash) DO UPDATE SET
                        original_name = EXCLUDED.original_name,
                        duration_seconds = EXCLUDED.duration_seconds,
                        speaker_count = EXCLUDED.speaker_count,
                        overall_confidence = EXCLUDED.overall_confidence,
                        needs_human_review = EXCLUDED.needs_human_review,
                        result_json = EXCLUDED.result_json
                    RETURNING id
                """),
                {
                    "audio_hash": audio_hash,
                    "original_name": original_name,
                    "duration_seconds": duration_seconds,
                    "speaker_count": speaker_count,
                    "overall_confidence": overall_confidence,
                    "needs_human_review": needs_human_review,
                    "result_json": json.dumps(result_json),
                },
            )
            row = result.first()
            if row:
                logger.info("Analysis saved to DB (id=%s, hash=%s...)", row[0], audio_hash[:12])
                return row[0]
    except Exception as exc:
        logger.error("Failed to save analysis to DB: %s", exc)
    return None