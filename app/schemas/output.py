"""Output schema: the 6 required fields per speaker.

Matches the exact schema from designdoc.md §4.5.
"""

from typing import Optional
from pydantic import BaseModel, Field


class EmotionOutput(BaseModel):
    primary: str
    secondary: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ToneOutput(BaseModel):
    label: str
    reasoning: str = ""


class AbuseOutput(BaseModel):
    flagged: bool = False
    category: str = "none"
    severity_1_to_5: int = 0
    evidence_span: Optional[str] = None
    reasoning: str = ""


class EnergyLoudnessOutput(BaseModel):
    level: str = "normal"
    description: str = ""
    dsp_agrees: bool = True


class Confidence(BaseModel):
    overall: float = 0.0
    needs_human_review: bool = False


class SpeakerResult(BaseModel):
    speaker_id: str
    segment_time_range: dict = Field(default_factory=dict)
    transcript: str = ""
    emotion: EmotionOutput = Field(default_factory=EmotionOutput)
    tone: ToneOutput = Field(default_factory=ToneOutput)
    abuse: AbuseOutput = Field(default_factory=AbuseOutput)
    energy_loudness: EnergyLoudnessOutput = Field(default_factory=EnergyLoudnessOutput)
    summary: str = ""
    confidence: Confidence = Field(default_factory=Confidence)


class AnalysisResponse(BaseModel):
    status: str = "ok"
    audio_hash: str = ""
    duration_seconds: float = 0.0
    speaker_count: int = 0
    results: list[SpeakerResult] = Field(default_factory=list)
    overall_confidence: float = 0.0
    needs_human_review: bool = False