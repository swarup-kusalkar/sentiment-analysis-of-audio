"""Tests for the guardrail module (Phase 7)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("NVIDIA_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")

from app.pipeline.guardrail import guardrail
from app.schemas.output import (
    SpeakerResult,
    EmotionOutput,
    ToneOutput,
    AbuseOutput,
    EnergyLoudnessOutput,
    Confidence,
)
from app.schemas.internal import DSPResult


def _dsp(bucket: str, f0_var=None, high_arousal=False):
    return DSPResult(
        rms_dbfs=-30.0,
        loudness_bucket=bucket,
        f0_variance=f0_var,
        high_arousal=high_arousal,
    )


def _speaker(
    energy_level="normal",
    energy_desc="",
    abuse_flagged=False,
    abuse_category="none",
    abuse_severity=0,
    abuse_evidence=None,
    abuse_reasoning="",
    emotion_primary="neutral",
    emotion_secondary=None,
    emotion_reasoning="",
    llm_confidence=0.85,
    transcript="",
):
    return SpeakerResult(
        speaker_id="SPEAKER_01",
        segment_time_range={"start_s": 0.0, "end_s": 10.0},
        transcript=transcript,
        emotion=EmotionOutput(
            primary=emotion_primary,
            secondary=emotion_secondary or [],
            reasoning=emotion_reasoning,
        ),
        tone=ToneOutput(label="calm", reasoning=""),
        abuse=AbuseOutput(
            flagged=abuse_flagged,
            category=abuse_category,
            severity_1_to_5=abuse_severity,
            evidence_span=abuse_evidence,
            reasoning=abuse_reasoning,
        ),
        energy_loudness=EnergyLoudnessOutput(
            level=energy_level,
            description=energy_desc,
            dsp_agrees=True,
        ),
        summary="test speaker",
        confidence=Confidence(overall=llm_confidence, needs_human_review=False),
    )


class TestEnergyLoudnessReconciliation:
    def test_agreement(self):
        spk = _speaker(energy_level="normal")
        dsp = _dsp("normal")
        result = guardrail(spk, dsp)
        assert result.energy_loudness.dsp_agrees is True
        assert result.confidence.overall > 0.8

    def test_minor_disagreement(self):
        spk = _speaker(energy_level="loud")
        dsp = _dsp("normal")
        result = guardrail(spk, dsp)
        assert result.energy_loudness.dsp_agrees is True

    def test_major_disagreement(self):
        spk = _speaker(energy_level="shouting")
        dsp = _dsp("whisper")
        result = guardrail(spk, dsp)
        assert result.energy_loudness.dsp_agrees is False
        assert result.confidence.needs_human_review is True

    def test_moderate_disagreement(self):
        spk = _speaker(energy_level="loud")
        dsp = _dsp("whisper")
        result = guardrail(spk, dsp)
        assert result.energy_loudness.dsp_agrees is False

    def test_confidence_floor(self):
        spk = _speaker(energy_level="shouting", llm_confidence=0.9)
        dsp = _dsp("whisper")
        result = guardrail(spk, dsp)
        assert result.confidence.overall >= 0.0
        assert result.confidence.overall <= 1.0


class TestF0ArousalPenalty:
    def test_high_arousal_vs_neutral(self):
        spk = _speaker(emotion_primary="neutral")
        dsp = _dsp("normal", f0_variance=0.3, high_arousal=True)
        result = guardrail(spk, dsp)
        assert result.confidence.overall < 0.85

    def test_low_arousal_vs_angry(self):
        spk = _speaker(emotion_primary="angry")
        dsp = _dsp("normal", f0_variance=0.05, high_arousal=False)
        result = guardrail(spk, dsp)
        assert result.confidence.overall < 0.85

    def test_matching_arousal(self):
        spk = _speaker(emotion_primary="sad")
        dsp = _dsp("normal", f0_variance=0.05, high_arousal=False)
        result = guardrail(spk, dsp)
        assert result.confidence.overall == 0.85

    def test_no_f0_data(self):
        spk = _speaker(emotion_primary="angry")
        dsp = _dsp("normal", f0_variance=None, high_arousal=False)
        result = guardrail(spk, dsp)
        assert result.confidence.overall == 0.85  # no penalty


class TestIntegrationFullGuardrail:
    def test_multiple_penalties(self):
        # Major loudness mismatch + high arousal/neutral conflict
        spk = _speaker(emotion_primary="neutral")
        dsp = _dsp("shouting", f0_variance=0.3, high_arousal=True)
        result = guardrail(spk, dsp)
        assert result.confidence.overall < 0.85  # Multiple penalties
        assert result.confidence.needs_human_review is True

    def test_confidence_raw_number_never_in_user_output(self):
        # The guardrail returns confidence values internally but never exposes dbfs/Hz
        spk = _speaker()
        dsp = _dsp("normal")
        result = guardrail(spk, dsp)
        raw = result.model_dump()
        assert "rms_dbfs" in repr(dsp), "DSPResult carries internal numbers"
        # The user-facing output does NOT contain dB, Hz, etc
        assert "rms_dbfs" not in str(raw)
        assert "dbfs" not in raw["emotion"]
        assert "Hz" not in str(raw)