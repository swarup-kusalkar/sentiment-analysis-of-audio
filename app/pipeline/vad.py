"""Voice Activity Detection via Silero VAD (Phase 2).

- Load Silero VAD model locally (no GPU required)
- Accept a 16 kHz mono WAV path, return speech timestamps
- Short-circuit when no speech is detected
"""

import logging

from app.schemas.internal import VADResult, VADSegment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_MIN_SPEECH_S = 0.25
_MIN_SILENCE_S = 0.4


def _load_model():
    try:
        from silero_vad import load_silero_vad
        return load_silero_vad()
    except ImportError:
        pass

    import torch
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )
    return model


def _read_audio(path: str):
    try:
        from silero_vad import read_audio
        return read_audio(path, sampling_rate=SAMPLE_RATE)
    except ImportError:
        pass

    import torch
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return torch.from_numpy(audio.copy())


def _get_speech_timestamps(audio, model, threshold: float = 0.5) -> list[dict]:
    try:
        from silero_vad import get_speech_timestamps
        return get_speech_timestamps(audio, model, sampling_rate=SAMPLE_RATE)
    except ImportError:
        pass

    import torch
    import numpy as np

    num_samples = audio.numel()
    speech_ts = []
    in_speech = False

    model.eval()
    with torch.no_grad():
        for chunk_start in range(0, num_samples, 512):
            chunk = audio[chunk_start: chunk_start + 512]
            if chunk.numel() < 512:
                padding = 512 - chunk.numel()
                chunk = np.pad(chunk.numpy(), (0, padding)).copy()
                chunk = torch.from_numpy(chunk)
            speech_prob = model(chunk, SAMPLE_RATE).item()

            if speech_prob >= threshold:
                if not in_speech:
                    in_speech = True
                    speech_ts.append({"start": chunk_start, "end": chunk_start + 512})
                else:
                    speech_ts[-1]["end"] = chunk_start + 512
            else:
                if in_speech:
                    in_speech = False

    return speech_ts


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
    model = _load_model()
    audio = _read_audio(normalised_wav_path)
    if audio.numel() == 0:
        logger.warning("Empty audio passed to VAD — returning no speech")
        return VADResult(has_speech=False, segments=[], speech_duration_s=0.0)

    raw_segments = _get_speech_timestamps(audio, model)

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