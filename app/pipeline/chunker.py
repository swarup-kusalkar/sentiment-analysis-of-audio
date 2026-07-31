"""Speaker-boundary-aware audio chunking (Phase 4).

- Uses diarization timeline to slice audio into chunks
- Never crosses speaker boundaries
- Respects 60–120 s chunk duration limits
- Splits long same-speaker segments at VAD silence gaps
- Extracts per-chunk WAV files via ffmpeg to temp dir
"""

import logging
import math
import os
import subprocess
import uuid
from typing import Optional

from app.config import settings
from app.schemas.internal import ChunkInfo, DiarizationSegment, VADSegment

logger = logging.getLogger(__name__)


def _ensure_temp_dir() -> None:
    os.makedirs(settings.temp_dir, exist_ok=True)


def _export_wav(
    source_path: str,
    speaker_id: str,
    label: str,
    start_s: float,
    end_s: float,
) -> str:
    """Extract [start_s, end_s] from the source WAV into a chunk file."""
    _ensure_temp_dir()
    safe_speaker = speaker_id.replace("/", "_").replace(" ", "_")
    fname = f"{uuid.uuid4().hex}_{safe_speaker}_{label}.wav"
    out_path = os.path.join(settings.temp_dir, fname)

    cmd = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-ss", str(start_s),
        "-to", str(end_s),
        "-c:a", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        out_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        result.check_returncode()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg extraction failed [{start_s:.1f}s–{end_s:.1f}s]: "
            f"{exc.stderr.strip()}"
        ) from exc
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not installed — cannot extract chunks")

    logger.info("Exported %s %.1fs–%.1fs → %s", label, start_s, end_s, fname)
    return out_path


def _find_split_point(
    target_s: float,
    start_s: float,
    end_s: float,
    vad: list[VADSegment],
    tolerance_s: float = 5.0,
) -> float:
    lo = max(start_s, target_s - tolerance_s)
    hi = min(end_s, target_s + tolerance_s)

    candidates: list[float] = []
    prev_end = start_s
    for seg in vad:
        if prev_end < seg.start_s:
            mid = (prev_end + seg.start_s) / 2
            if lo <= mid <= hi:
                candidates.append(mid)
        prev_end = max(prev_end, seg.end_s)
    if prev_end < end_s:
        mid = (prev_end + end_s) / 2
        if lo <= mid <= hi:
            candidates.append(mid)

    if candidates:
        candidates.sort(key=lambda x: abs(x - target_s))
        return candidates[0]
    return target_s


def _split_range(
    start: float,
    end: float,
    max_dur: float,
    vad: list[VADSegment],
) -> list[tuple[float, float]]:
    if end - start <= max_dur:
        return [(round(start, 2), round(end, 2))]

    n = math.ceil((end - start) / max_dur)
    step = (end - start) / n

    cut_points = [start]
    for i in range(1, n):
        target = start + i * step
        if vad:
            split = _find_split_point(target, start, end, vad)
        else:
            split = target
        cut_points.append(split)
    cut_points.append(end)

    return [
        (round(cut_points[i], 2), round(cut_points[i + 1], 2))
        for i in range(len(cut_points) - 1)
    ]


def chunk(
    normalised_wav_path: str,
    diarization_segments: list[DiarizationSegment],
    vad_segments: Optional[list[VADSegment]] = None,
    max_duration_s: Optional[float] = None,
) -> list[ChunkInfo]:
    if not diarization_segments:
        logger.warning("No diarization segments — cannot create chunks")
        return []

    max_dur = max_duration_s or settings.chunk_max_duration_s
    vad = vad_segments or []

    # Sort diarization segments by start time
    sorted_segs = sorted(diarization_segments, key=lambda s: (s.start_s, s.end_s))

    # Collect all turns per speaker, maintaining original order
    speaker_turns: dict[str, list[tuple[float, float]]] = {}
    for seg in sorted_segs:
        speaker_turns.setdefault(seg.speaker_id, []).append((seg.start_s, seg.end_s))

    chunks: list[ChunkInfo] = []
    chunk_id = 0

    for speaker_id, turns in speaker_turns.items():
        # Merge close gaps within same speaker (< 0.3s)
        merged: list[tuple[float, float]] = []
        cur_s, cur_e = turns[0]
        for t_s, t_e in turns[1:]:
            if t_s - cur_e <= 0.3:
                cur_e = t_e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = t_s, t_e
        merged.append((cur_s, cur_e))

        for run_s, run_e in merged:
            for piece_s, piece_e in _split_range(run_s, run_e, max_dur, vad):
                label = f"{chunk_id:04d}"
                wav_path = _export_wav(
                    source_path=normalised_wav_path,
                    speaker_id=speaker_id,
                    label=label,
                    start_s=piece_s,
                    end_s=piece_e,
                )
                chunks.append(
                    ChunkInfo(
                        chunk_id=label,
                        speaker_id=speaker_id,
                        start_s=round(piece_s, 2),
                        end_s=round(piece_e, 2),
                        wav_path=wav_path,
                    )
                )
                chunk_id += 1

    duration_covered = sum(c.end_s - c.start_s for c in chunks)
    logger.info(
        "Chunking complete: %d chunks across %d speaker(s) (%.1f total seconds)",
        len(chunks),
        len(speaker_turns),
        duration_covered,
    )
    return chunks