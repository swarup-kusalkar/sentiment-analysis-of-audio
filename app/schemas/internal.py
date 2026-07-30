"""Internal pipeline schemas (Diarization segment, DSP result, VAD result)."""

from pydantic import BaseModel


class VADSegment(BaseModel):
    start_s: float
    end_s: float


class VADResult(BaseModel):
    has_speech: bool
    segments: list[VADSegment] = []
    speech_duration_s: float = 0.0


class DiarizationSegment(BaseModel):
    speaker_id: str
    start_s: float
    end_s: float


class DSPResult(BaseModel):
    rms_dbfs: float
    loudness_bucket: str
    f0_variance: float | None = None
    high_arousal: bool = False


class ChunkInfo(BaseModel):
    chunk_id: str
    speaker_id: str
    start_s: float
    end_s: float
    wav_path: str