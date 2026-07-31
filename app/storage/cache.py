"""Redis caching for audio hash and diarization results.

Avoids re-processing identical audio files and re-running diarization on known content (Phase 8).
"""

import json
from typing import Optional


async def get_diarization(audio_hash: str) -> Optional[list]:
    return None


async def set_diarization(audio_hash: str, segments: list) -> None:
    pass