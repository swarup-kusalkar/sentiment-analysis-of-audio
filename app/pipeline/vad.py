"""Voice Activity Detection via Silero VAD (Phase 2).

- Load Silero VAD model locally (no GPU required)
- Accept a 16 kHz mono WAV path, return speech timestamps
- Short-circuit when no speech is detected
"""

import logging

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

from app.schemas.internal import VADResult, VADSegment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_MIN_SPEECH_S = 0.25
_MIN_SILENCE_S = 0.4


def _merge_intervals(segments: list[VADSegment], gap_s: float) -> list[VADSegment]:
    if not segments:
        return []
    merged = [segments[0]]
    for cur in segments[1:]:
        prev = merged[-1]
        if cur.start_s - prev.end_s <= gap_s:
            prev.end_s = cur.end_s
        else:
            merged.append(cur)
    return merged


def vad(normalised_wav_path: str) -> VADResult:
    model = load_silero_vad()
    audio = read_audio(normalised_wav_path, sampling_rate=SAMPLE_RATE)

    if audio.numel() == 0:
        logger.warning("Empty audio passed to VAD — returning no speech")
        return VADResult(has_speech=False, segments=[], speech_duration_s=0.0)

    raw_segments = get_speech_timestamps(audio, model, sampling_rate=SAMPLE_RATE)

    segments = []
    for seg in raw_segments:
        start_s = seg["start"] / SAMPLE_RATE
        end_s = seg["end"] / SAMPLE_RATE
        duration = end_s - start_s
        if duration >= _MIN_SPEECH_S:
            segments.append(VADSegment(start_s=start_s, end_s=end_s))

    if not segments:
        logger.info("No speech segments detected in audio file")
        return VADResult(has_speech=False, segments=[], speech_duration_s=0.0)

    segments = _merge_intervals(segments, gap_s=_MIN_SILENCE_S)

    total_speech = sum(seg.end_s - seg.start_s for seg in segments)
    logger.info(
        "VAD detected %d speech segment(s), %.1fs total speech",
        len(segments),
        total_speech,
    )
    return VADResult(
        has_speech=total_speech > 0,
        segments=segments,
        speech_duration_s=round(total_speech, 2),
    )
