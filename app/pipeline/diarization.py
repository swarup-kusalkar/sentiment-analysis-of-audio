"""Speaker diarization via pyannote.audio (Phase 3).

- Identify who spoke when (speaker_id, start_s, end_s)
- Returns RTTM-style timeline
- Results cached in Redis by audio hash
"""

import logging

from pyannote.audio import Pipeline

from app.config import settings
from app.schemas.internal import DiarizationSegment
from app.storage.cache import get_diarization, set_diarization

logger = logging.getLogger(__name__)


def _get_pipeline() -> Pipeline:
    import torch

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


def _segments_from_annotation(annotation) -> list[DiarizationSegment]:
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
    merged: list[DiarizationSegment] = []
    for cur in segments:
        if (
            merged
            and merged[-1].speaker_id == cur.speaker_id
            and cur.start_s - merged[-1].end_s <= gap_s
        ):
            merged[-1].end_s = cur.end_s
        else:
            merged.append(cur)
    return merged


async def diarize(normalised_wav_path: str, audio_hash: str) -> list[DiarizationSegment]:
    cached = await get_diarization(audio_hash)
    if cached:
        logger.info("Diarization cache hit for %s", audio_hash[:12])
        return cached

    pipeline = _get_pipeline()
    diarization = pipeline(normalised_wav_path)

    segments = _merge_adjacent(_segments_from_annotation(diarization), gap_s=0.1)

    if not segments:
        segments = [
            DiarizationSegment(speaker_id="SPEAKER_00", start_s=0.0, end_s=0.0)
        ]
        logger.warning("No speakers found by diarization — using a single SPEAKER_00 segment")

    speaker_count = len({s.speaker_id for s in segments})
    logger.info(
        "Diarization complete: %d speaker(s), %d segment(s)",
        speaker_count,
        len(segments),
    )

    await set_diarization(audio_hash, segments)
    return segments
