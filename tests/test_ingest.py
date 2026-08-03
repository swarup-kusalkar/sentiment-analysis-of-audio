"""Tests for the ingest module (Phase 1)."""

import hashlib
import math
import os
import struct
import sys
import tempfile
import wave

import pytest

# Patch settings before importing ingest to avoid needing .env
os.environ.setdefault("NVIDIA_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline.ingest import (
    _normalise,
    _compute_content_hash,
    _get_probe,
    IngestError,
    AudioFile,
)


def _create_test_wav(duration_s=2.0, sample_rate=44100, channels=2, freq=440):
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="test_")
    os.close(fd)
    n_samples = int(sample_rate * duration_s)
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n_samples):
            val = int(16000 * math.sin(2 * math.pi * freq * i / sample_rate))
            w.writeframes(struct.pack("<h", max(-32768, min(32767, val))))
    return path


class TestNormalise:
    """Tests for _normalise — ffmpeg conversion to 16kHz mono WAV."""

    def test_normalise_stereo_to_mono(self):
        raw = _create_test_wav(duration_s=2.0, sample_rate=44100, channels=2)
        try:
            result = _normalise(raw)
            assert result.endswith(".wav"), "Output should be .wav"
            assert os.path.exists(result), "Normalised file should exist"
            assert os.path.getsize(result) > 0, "Normalised file should not be empty"
        finally:
            for p in [raw, result]:
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_normalise_mono_already(self):
        raw = _create_test_wav(duration_s=1.5, sample_rate=16000, channels=1)
        try:
            result = _normalise(raw)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
        finally:
            for p in [raw, result]:
                try:
                    os.remove(p)
                except OSError:
                    pass

    def test_normalise_rejects_missing_file(self):
        try:
            _normalise("/tmp/nonexistent_audio_file.wav")
            assert False, "Should have raised IngestError"
        except IngestError:
            pass


class TestComputeContentHash:
    def test_hash_is_consistent(self):
        path = _create_test_wav(duration_s=0.5, sample_rate=16000, channels=1)
        try:
            h1 = _compute_content_hash(path)
            h2 = _compute_content_hash(path)
            assert h1 == h2, "Same file should produce same hash"
            assert len(h1) == 64, "SHA256 hex digest is 64 chars"
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_different_files_have_different_hashes(self):
        path1 = _create_test_wav(duration_s=1.0)
        path2 = _create_test_wav(duration_s=2.0)
        try:
            h1 = _compute_content_hash(path1)
            h2 = _compute_content_hash(path2)
            assert h1 != h2, "Different files should have different hashes"
        finally:
            for p in [path1, path2]:
                try:
                    os.remove(p)
                except OSError:
                    pass


class TestGetProbe:
    def test_probe_valid_wav(self):
        path = _create_test_wav(duration_s=2.5, sample_rate=16000, channels=1)
        try:
            info = _get_probe(path)
            assert "duration" in info
            assert info["channels"] == 1
            assert info["sample_rate"] == 16000
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_probe_stereo_wav(self):
        path = _create_test_wav(duration_s=2.5, sample_rate=16000, channels=2)
        try:
            info = _get_probe(path)
            assert info["channels"] == 2
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def test_probe_non_existent(self):
        try:
            _get_probe("/tmp/nonexistent.wav")
            assert False, "Should have raised"
        except IngestError:
            pass


class TestAudioFile:
    def test_audio_file_creation(self):
        af = AudioFile(
            original_name="test.wav",
            normalised_path="/tmp/test.wav",
            content_hash="abc123",
            duration_s=10.0,
            sample_rate=16000,
            channels=1,
        )
        assert af.original_name == "test.wav"
        assert af.duration_s == 10.0
        assert af.sample_rate == 16000
        assert "test.wav" in repr(af)