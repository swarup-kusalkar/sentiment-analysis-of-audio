"""Tests for the DSP module (Phase 5)."""

import math
import os
import struct
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("NVIDIA_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")

from app.pipeline.dsp import _rms_to_loudness_bucket, _compute_rms_dbfs
from app.schemas.internal import DSPResult


def _create_test_wav(duration_s=2.0, amplitude=0.5, freq=440):
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="test_dsp_")
    os.close(fd)
    sr = 16000
    n = int(sr * duration_s)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            val = int(amplitude * 32767 * math.sin(2 * math.pi * freq * i / sr))
            w.writeframes(struct.pack("<h", max(-32768, min(32767, val))))
    return path


class TestRMSComputation:
    def test_silence(self):
        y = np.zeros(16000, dtype=np.float32)
        dbfs = _compute_rms_dbfs(y)
        assert dbfs < -100, f"Silence should give very low dBFS, got {dbfs}"
        bucket = _rms_to_loudness_bucket(dbfs)
        assert bucket == "whisper", f"Silence should be whisper, got {bucket}"

    def test_fullscale_should_be_loud(self):
        y = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.9
        dbfs = _compute_rms_dbfs(y)
        bucket = _rms_to_loudness_bucket(dbfs)
        assert bucket in ("loud", "shouting"), f"Full scale should be at least loud, got {bucket}"

    def test_very_quiet(self):
        y = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.05
        dbfs = _compute_rms_dbfs(y)
        bucket = _rms_to_loudness_bucket(dbfs)
        assert bucket in ("whisper", "quiet"), f"Quiet signal should be QUIET/WHISPER, got {bucket}"

    def test_medium(self):
        y = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.25
        dbfs = _compute_rms_dbfs(y)
        bucket = _rms_to_loudness_bucket(dbfs)
        assert bucket in ("normal", "loud"), f"Got bucket '{bucket}'"


class TestBucketThresholds:
    def test_all_threshold_edges(self):
        # Explicit boundary checks via mocked thresholds
        test_values = [
            (-45.0, "whisper"),
            (-37.0, "quiet"),
            (-25.0, "normal"),
            (-15.0, "loud"),
            (-5.0, "shouting"),
        ]
        for dbfs, expected in test_values:
            # Use the actual function with default config thresholds [-40, -30, -20, -10]
            bucket = _rms_to_loudness_bucket(dbfs)
            assert bucket == expected, f"{dbfs}dB expected {expected}, got {bucket}"


class TestDSPResult:
    def test_dsp_result_fields(self):
        r = DSPResult(
            rms_dbfs=-15.0,
            loudness_bucket="loud",
            f0_variance=0.2,
            high_arousal=True,
        )
        assert r.loudness_bucket == "loud"
        assert r.high_arousal is True
        assert r.rms_dbfs == -15.0
        assert r.f0_variance == 0.2

    def test_dsp_no_f0(self):
        r = DSPResult(
            rms_dbfs=-30.0,
            loudness_bucket="quiet",
            f0_variance=None,
            high_arousal=False,
        )
        assert r.f0_variance is None
        assert r.high_arousal is False