"""Speaker diarization via pyannote.audio (Phase 3).

- Identify who spoke when (speaker_id, start_s, end_s)
- Returns RTTM-style timeline
- Results cached in Redis by audio hash
"""

import logging
from typing import Optional

from app.schemas.internal import DiarizationSegment

logger = logging.getLogger(__name__)


def _get_pipeline():
    import torch
    from pyannote.audio import Pipeline

    from app.config import settings

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=settings.hf_token,
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        logger.info("Diarization pipeline moved to GPU")
    else:
        logger.info("Diarization pipeline running on CPU")

    return pipeline


async def _cache_get(audio_hash: str) -> Optional[list[DiarizationSegment]]:
    try:
        from app.storage.cache import get_diarization
        return await get_diarization(audio_hash)
    except Exception:
        return None


async def _cache_set(
    audio_hash: str, segments: list[DiarizationSegment]
) -> None:
    try:
        from app.storage.cache import set_diarization
        await set_diarization(audio_hash, segments)
    except Exception:
        pass


def _infer_num_speakers(segments: list[DiarizationSegment]) -> int:
    speakers = {s.speaker_id for s in segments}
    return len(speakers)


def _segments_from_annotation(annotation) -> list[DiarizationSegment]:
    """Convert pyannote.core.Annotation to list of DiarizationSegment."""
    segments = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                speaker_id=speaker,
                start_s=round(segment.start, 2),
                end_s=round(segment.end, 2),
            )
        )
    return segments


def _merge_adjacent(segments: list[DiarizationSegment], gap_s: float = 0.0) -> list[DiarizationSegment]:
    """Merge adjacent segments from the same speaker separated by <= gap_s."""
    if not segments:
        return []
    merged = [segments[0]]
    for cur in segments[1:]:
        prev = merged[-1]
        if (
            cur.speaker_id == prev.speaker_id
            and cur.start_s - prev.end_s <= gap_s
        ):
            prev.end_s = cur.end_s
        else:
            merged.append(cur)
    return merged


async def diarize(
    normalised_wav_path: str,
    audio_hash: str,
    num_speakers: Optional[int] = None,
) -> list[DiarizationSegment]:
    # Try cache first
    cached = await _cache_get(audio_hash)
    if cached:
        logger.info("Diarization cache hit for %s", audio_hash[:12])
        return cached

    pipeline = _get_pipeline()

    diarization = pipeline(
        normalised_wav_path,
        num_speakers=num_speakers if num_speakers else None,
    )

    segments = _segments_from_annotation(diarization)
    segments = _merge_adjacent(segments, gap_s=0.1)

    if not segments:
        segments = [
            DiarizationSegment(
                speaker_id="SPEAKER_00",
                start_s=0.0,
                end_s=0.0,
            )
        ]
        logger.warning("No speakers found by diarization — using a single SPEAKER_00 segment")

    speaker_count = _infer_num_speakers(segments)
    logger.info(
        "Diarization complete: %d speaker(s), %d segment(s)",
        speaker_count,
        len(segments),
    )

    # Cache for future reuse
    await _cache_set(audio_hash, segments)

    return segments