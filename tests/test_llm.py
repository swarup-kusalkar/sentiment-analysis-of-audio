"""Tests for the LLM module (Phase 6)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("NVIDIA_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")

from app.pipeline.llm import _extract_json
from app.schemas.output import (
    SpeakerResult,
    EmotionOutput,
    ToneOutput,
    AbuseOutput,
    EnergyLoudnessOutput,
    Confidence,
)


class TestExtractJSON:
    def test_clean_json(self):
        raw = '{"transcript": "hello", "emotion": {"primary": "neutral"}}'
        extracted = _extract_json(raw)
        # Should be valid JSON
        parsed = json.loads(extracted)
        assert parsed["transcript"] == "hello"

    def test_json_with_backticks(self):
        raw = '```json\n{"transcript": "hello"}\n```'
        extracted = _extract_json(raw)
        parsed = json.loads(extracted)
        assert parsed["transcript"] == "hello"

    def test_json_with_prefix(self):
        raw = "Here is the result: {\"transcript\": \"hello\"}"
        extracted = _extract_json(raw)
        parsed = json.loads(extracted)
        assert parsed["transcript"] == "hello"

    def test_json_with_text_surrounding(self):
        raw = "Some thinking text... {\"transcript\": \"hello\"} End of result."
        extracted = _extract_json(raw)
        parsed = json.loads(extracted)
        assert parsed["transcript"] == "hello"

    def test_invalid_json(self):
        raw = "this is not json at all"
        extracted = _extract_json(raw)
        # Should return something (original text) but not crash
        assert isinstance(extracted, str)


class TestSpeakerResultValidation:
    def test_minimal_result(self):
        result = SpeakerResult(
            speaker_id="speaker_1",
            segment_time_range={"start_s": 0.0, "end_s": 10.0},
            transcript="Hello world",
            emotion=EmotionOutput(primary="neutral", secondary=[], reasoning="test"),
            tone=ToneOutput(label="calm", reasoning="test"),
            abuse=AbuseOutput(
                flagged=False,
                category="none",
                severity_1_to_5=0,
                evidence_span=None,
                reasoning="",
            ),
            energy_loudness=EnergyLoudnessOutput(
                level="normal",
                description="steady",
                dsp_agrees=True,
            ),
            summary="Test summary",
            confidence=Confidence(overall=0.85, needs_human_review=False),
        )
        assert result.speaker_id == "speaker_1"
        assert result.emotion.primary == "neutral"
        assert result.confidence.overall > 0.8

    def test_abusive_result(self):
        result = SpeakerResult(
            speaker_id="speaker_2",
            segment_time_range={"start_s": 0.0, "end_s": 5.0},
            transcript="Profanity example",
            emotion=EmotionOutput(
                primary="angry",
                secondary=["frustrated"],
                reasoning="loud speaker",
            ),
            tone=ToneOutput(label="aggressive", reasoning="forceful"),
            abuse=AbuseOutput(
                flagged=True,
                category="profanity",
                severity_1_to_5=3,
                evidence_span="bad word",
                reasoning="f-word",
            ),
            energy_loudness=EnergyLoudnessOutput(
                level="shouting",
                description="very loud",
                dsp_agrees=True,
            ),
            summary="Abusive speaker",
            confidence=Confidence(overall=0.7, needs_human_review=False),
        )
        assert result.abuse.flagged is True
        assert result.abuse.category == "profanity"
        assert result.abuse.severity_1_to_5 == 3
        assert result.abuse.evidence_span == "bad word"

    def test_default_abuse(self):
        output = AbuseOutput()
        assert output.flagged is False
        assert output.category == "none"
        assert output.severity_1_to_5 == 0

    def test_full_schema_round_trip(self):
        result = SpeakerResult(
            speaker_id="spk_1",
            segment_time_range={"start_s": 0.0, "end_s": 5.0},
            transcript="Hello",
            emotion=EmotionOutput(
                primary="happy",
                secondary=["excited"],
                reasoning="",
            ),
            tone=ToneOutput(label="friendly", reasoning=""),
            abuse=AbuseOutput(
                flagged=True,
                category="threat",
                severity_1_to_5=4,
                evidence_span="I will hurt you",
                reasoning="",
            ),
            energy_loudness=EnergyLoudnessOutput(
                level="shouting",
                description="very loud descriptive",
                dsp_agrees=True,
            ),
            summary="Happy speaker says hi",
            confidence=Confidence(overall=0.9, needs_human_review=False),
        )
        data = result.model_dump()
        assert data["speaker_id"] == "spk_1"
        assert data["abuse"]["flagged"] is True
        assert data["abuse"]["category"] == "threat"
        assert data["abuse"]["evidence_span"] == "I will hurt you"
        assert data["abuse"]["severity_1_to_5"] == 4
        assert data["energy_loudness"]["level"] == "shouting"
        assert data["energy_loudness"]["dsp_agrees"] is True
        assert data["emotion"]["primary"] == "happy"
        assert "excited" in data["emotion"]["secondary"]