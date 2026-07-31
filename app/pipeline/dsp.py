"""DSP grounding: energy/loudness + F0 variance (Phase 5).

- RMS energy (dBFS) → loudness bucket (whisper…shouting)
- F0 variance → soft arousal signal (never exposed to user)
- Both computed in one audio pass; raw dBFS/F0 numbers are strictly internal
"""

import logging

import numpy as np

from app.config import settings
from app.schemas.internal import DSPResult

logger = logging.getLogger(__name__)


def _rms_to_loudness_bucket(rms_dbfs: float) -> str:
    if rms_dbfs <= settings.dsp_whisper_threshold:
        return "whisper"
    if rms_dbfs <= settings.dsp_quiet_threshold:
        return "quiet"
    if rms_dbfs <= settings.dsp_normal_threshold:
        return "normal"
    if rms_dbfs <= settings.dsp_loud_threshold:
        return "loud"
    return "shouting"


def _compute_rms_dbfs(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(y ** 2)))
    return float(20 * np.log10(rms + 1e-12))


def _compute_f0_variance(wav_path: str) -> tuple[float | None, bool]:
    try:
        import parselmouth

        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch(
            time_step=0.01,
            pitch_floor=75.0,
            pitch_ceiling=600.0,
        )
        values = pitch.selected_array["frequency"]
        values = values[values > 0]

        if len(values) < 3:
            return None, False

        norm_var = float(np.var(values) / (np.mean(values) + 1e-9))
        return round(norm_var, 4), norm_var > 0.15

    except Exception:
        pass

    try:
        import librosa

        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            fill_na=0.0,
        )
        f0 = f0[f0 > 0]
        if len(f0) < 3:
            return None, False

        norm_var = float(np.var(f0) / (np.mean(f0) + 1e-9))
        return round(norm_var, 4), norm_var > 0.15

    except Exception:
        return None, False


def analyse_chunk(wav_path: str) -> DSPResult:
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)

    if y.size == 0:
        logger.warning("Empty audio chunk: %s", wav_path)
        return DSPResult(
            rms_dbfs=-100.0,
            loudness_bucket="whisper",
            f0_variance=None,
            high_arousal=False,
        )

    rms_dbfs = _compute_rms_dbfs(y)
    bucket = _rms_to_loudness_bucket(rms_dbfs)

    f0_var: float | None = None
    high_arousal = False

    result = _compute_f0_variance(wav_path)
    if result[0] is not None:
        f0_var = result[0]
        high_arousal = result[1]

    logger.info(
        "DSP: %.1f dBFS → %s | arousal=%s",
        rms_dbfs,
        bucket,
        high_arousal,
    )

    return DSPResult(
        rms_dbfs=round(rms_dbfs, 1),
        loudness_bucket=bucket,
        f0_variance=f0_var,
        high_arousal=high_arousal,
    )