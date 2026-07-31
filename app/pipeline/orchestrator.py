"""Pipeline execution coordinator (Phase 8).

Wires Phases 1–7 together and returns final per-speaker results.
"""

import asyncio
import logging
import os
from typing import Optional

from app.config import settings
from app.pipeline.ingest import AudioFile, ingest
from app.pipeline.vad import vad, VADResult
from app.pipeline.diarization import diarize, DiarizationSegment
from app.pipeline.chunker import chunk, ChunkInfo
from app.pipeline.dsp import analyse_chunk as dsp_analyse_chunk, DSPResult
from app.pipeline.llm import analyse_chunk as llm_analyse_chunk, LLMError
from app.pipeline.guardrail import guardrail
from app.schemas.output import (
    SpeakerResult,
    AnalysisResponse,
    EmotionOutput,
    ToneOutput,
    AbuseOutput,
    EnergyLoudnessOutput,
    Confidence,
)
from app.storage.cache import get_analysis, set_analysis
from app.storage.db import save_analysis, init_db

logger = logging.getLogger(__name__)


async def process_audio(audio: AudioFile) -> AnalysisResponse:
    """Full pipeline: VAD -> Diarization -> Chunking -> per-chunk (DSP+LLM+Guardrail) -> Aggregate."""

    # 1. VAD - short-circuit if no speech
    vad_result: VADResult = vad(audio.normalised_path)
    if not vad_result.has_speech:
        logger.info("VAD detected no speech; returning empty result")
        return AnalysisResponse(
            status="no_speech",
            audio_hash=audio.content_hash,
            duration_seconds=round(audio.duration_s, 2),
            speaker_count=0,
            results=[],
            overall_confidence=0.0,
            needs_human_review=False,
        )

    # 2. Diarization (cached by audio_hash)
    diarization_segments = await diarize(
        normalised_wav_path=audio.normalised_path,
        audio_hash=audio.content_hash,
    )

    # 3. Chunking - speaker-aware
    chunk_infos: list[ChunkInfo] = chunk(
        normalised_wav_path=audio.normalised_path,
        diarization_segments=diarization_segments,
        vad_segments=vad_result.segments,
    )

    if not chunk_infos:
        logger.warning("No chunks produced from diarization")
        return AnalysisResponse(
            status="error",
            audio_hash=audio.content_hash,
            duration_seconds=round(audio.duration_s, 2),
            speaker_count=0,
            results=[],
            overall_confidence=0.0,
            needs_human_review=True,
        )

    # 4. Process each chunk through DSP + LLM + Guardrail
    speaker_chunks: dict[str, list[ChunkInfo]] = {}
    for ci in chunk_infos:
        speaker_chunks.setdefault(ci.speaker_id, []).append(ci)

    speaker_results: dict[str, SpeakerResult] = {}

    for speaker_id, chunks in speaker_chunks.items():
        logger.info("Processing %d chunk(s) for %s", len(chunks), speaker_id)

        merged_transcript_parts = []
        merged_emotions = []
        merged_tones = []
        merged_abuses = []
        merged_energy = []
        merged_summaries = []
        chunk_confidences = []
        needs_review_any = False

        for ci in chunks:
            try:
                # DSP
                dsp_result = dsp_analyse_chunk(ci.wav_path)

                # LLM
                llm_result = await llm_analyse_chunk(
                    wav_path=ci.wav_path,
                    speaker_id=ci.speaker_id,
                    start_s=ci.start_s,
                    end_s=ci.end_s,
                )

                # Guardrail
                guarded = guardrail(llm_result, dsp_result)

                # Collect for merge
                merged_transcript_parts.append(guarded.transcript)
                merged_emotions.append(guarded.emotion)
                merged_tones.append(guarded.tone)
                merged_abuses.append(guarded.abuse)
                merged_energy.append(guarded.energy_loudness)
                merged_summaries.append(guarded.summary)
                chunk_confidences.append(guarded.confidence.overall)
                if guarded.confidence.needs_human_review:
                    needs_review_any = True

            except LLMError as exc:
                logger.error("LLM failed for chunk %s: %s", ci.chunk_id, exc)
                # Create a fallback result
                fallback = _fallback_speaker_result(ci, str(exc))
                merged_transcript_parts.append(fallback.transcript)
                merged_emotions.append(fallback.emotion)
                merged_tones.append(fallback.tone)
                merged_abuses.append(fallback.abuse)
                merged_energy.append(fallback.energy_loudness)
                merged_summaries.append(fallback.summary)
                chunk_confidences.append(fallback.confidence.overall)
                needs_review_any = True

        # Merge per-speaker results
        speaker_results[speaker_id] = _merge_speaker_chunks(
            speaker_id=speaker_id,
            chunks=chunks,
            transcripts=merged_transcript_parts,
            emotions=merged_emotions,
            tones=merged_tones,
            abuses=merged_abuses,
            energy_list=merged_energy,
            summaries=merged_summaries,
            confidences=chunk_confidences,
            needs_review=needs_review_any,
        )

    # 5. Build final response
    results_list = list(speaker_results.values())
    overall_confidence = (
        sum(r.confidence.overall for r in results_list) / len(results_list)
        if results_list else 0.0
    )
    needs_human_review_any = any(r.confidence.needs_human_review for r in results_list)

    response = AnalysisResponse(
        status="ok",
        audio_hash=audio.content_hash,
        duration_seconds=round(audio.duration_s, 2),
        speaker_count=len(results_list),
        results=results_list,
        overall_confidence=round(overall_confidence, 2),
        needs_human_review=needs_human_review_any,
    )

    # 6. Persist to DB and cache
    await _persist_results(audio, response)

    return response


def _merge_speaker_chunks(
    speaker_id: str,
    chunks: list[ChunkInfo],
    transcripts: list[str],
    emotions: list[EmotionOutput],
    tones: list[ToneOutput],
    abuses: list[AbuseOutput],
    energy_list: list[EnergyLoudnessOutput],
    summaries: list[str],
    confidences: list[float],
    needs_review: bool,
) -> SpeakerResult:
    # Full transcript with [timestamp] markers
    full_transcript = " ".join(transcripts)

    # Emotion: majority vote on primary, collect unique secondaries
    from collections import Counter
    primary_counts = Counter(e.primary for e in emotions)
    primary_emotion = primary_counts.most_common(1)[0][0] if primary_counts else "unknown"
    all_secondaries = []
    for e in emotions:
        all_secondaries.extend(e.secondary)
    unique_secondaries = list(dict.fromkeys(all_secondaries))

    # Tone: majority vote
    tone_counts = Counter(t.label for t in tones)
    primary_tone = tone_counts.most_common(1)[0][0] if tone_counts else "unknown"

    # Abuse: flag if any chunk flagged
    flagged_abuse = any(a.flagged for a in abuses)
    abuse_category = "none"
    max_severity = 0
    evidence_spans = []
    abuse_reasons = []
    if flagged_abuse:
        # Find the most severe abuse
        for a in abuses:
            if a.flagged:
                if a.severity_1_to_5 > max_severity:
                    max_severity = a.severity_1_to_5
                    abuse_category = a.category
                if a.evidence_span:
                    evidence_spans.append(a.evidence_span)
                if a.reasoning:
                    abuse_reasons.append(a.reasoning)

    # Energy: majority vote on level, combine descriptions
    energy_counts = Counter(e.level for e in energy_list)
    primary_energy = energy_counts.most_common(1)[0][0] if energy_counts else "normal"
    all_energy_desc = [e.description for e in energy_list if e.description]
    combined_energy_desc = " | ".join(all_energy_desc) if all_energy_desc else ""
    dsp_agrees_all = all(e.dsp_agrees for e in energy_list)

    # Summary: combine
    combined_summary = " ".join(summaries) if summaries else ""

    # Overall confidence
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return SpeakerResult(
        speaker_id=speaker_id,
        segment_time_range={
            "start_s": round(chunks[0].start_s, 2),
            "end_s": round(chunks[-1].end_s, 2),
        },
        transcript=full_transcript,
        emotion=EmotionOutput(
            primary=primary_emotion,
            secondary=unique_secondaries,
            reasoning=f"Merged from {len(chunks)} chunk(s); primary={primary_emotion}",
        ),
        tone=ToneOutput(
            label=primary_tone,
            reasoning=f"Merged from {len(chunks)} chunk(s)",
        ),
        abuse=AbuseOutput(
            flagged=flagged_abuse,
            category=abuse_category,
            severity_1_to_5=max_severity,
            evidence_span=" | ".join(evidence_spans) if evidence_spans else None,
            reasoning="; ".join(abuse_reasons) if abuse_reasons else "",
        ),
        energy_loudness=EnergyLoudnessOutput(
            level=primary_energy,
            description=combined_energy_desc,
            dsp_agrees=dsp_agrees_all,
        ),
        summary=combined_summary,
        confidence=Confidence(
            overall=round(avg_conf, 2),
            needs_human_review=needs_review,
        ),
    )


def _fallback_speaker_result(chunk: ChunkInfo, error: str) -> SpeakerResult:
    return SpeakerResult(
        speaker_id=chunk.speaker_id,
        segment_time_range={"start_s": chunk.start_s, "end_s": chunk.end_s},
        transcript=f"[LLM ERROR: {error}]",
        emotion=EmotionOutput(primary="unknown", secondary=[], reasoning=error),
        tone=ToneOutput(label="unknown", reasoning=error),
        abuse=AbuseOutput(flagged=False, category="none", severity_1_to_5=0, evidence_span=None, reasoning=""),
        energy_loudness=EnergyLoudnessOutput(level="normal", description="", dsp_agrees=True),
        summary=f"Processing failed: {error}",
        confidence=Confidence(overall=0.0, needs_human_review=True),
    )


async def _persist_results(audio: AudioFile, response: AnalysisResponse) -> None:
    # Initialize DB on first use
    await init_db()

    # Prepare result JSON for storage
    result_json = {
        "status": response.status,
        "audio_hash": response.audio_hash,
        "duration_seconds": response.duration_seconds,
        "speaker_count": response.speaker_count,
        "results": [r.model_dump() for r in response.results],
        "overall_confidence": response.overall_confidence,
        "needs_human_review": response.needs_human_review,
    }

    # Save to PostgreSQL (async, fire-and-forget-ish)
    asyncio.create_task(
        save_analysis(
            audio_hash=audio.content_hash,
            original_name=audio.original_name,
            duration_seconds=audio.duration_s,
            speaker_count=response.speaker_count,
            overall_confidence=response.overall_confidence,
            needs_human_review=response.needs_human_review,
            result_json=result_json,
        )
    )

    # Save to Redis cache
    asyncio.create_task(set_analysis(audio.content_hash, result_json))