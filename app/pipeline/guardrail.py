"""Merge and guardrail layer (Phase 7).

- Reconciles DSP loudness with LLM energy_loudness
- Lexicon cross-check (better-profanity vs. LLM abuse detection)
- Aggregates per-field and overall confidence
- Flags low-confidence or disagreement for human review
"""

import logging

from app.pipeline.dsp import DSPResult
from app.lexicon.checker import check_transcript
from app.schemas.output import SpeakerResult

logger = logging.getLogger(__name__)


_LOUDNESS_ORDER = {
    "whisper": 0,
    "quiet": 1,
    "normal": 2,
    "loud": 3,
    "shouting": 4,
}


def _reconcile_energy_loudness(
    dsp: DSPResult,
    llm_result: SpeakerResult,
) -> tuple[bool, float]:
    """Compare DSP loudness bucket vs LLM described level.

    Returns:
      (dsp_agrees: bool, penalty: float) — penalty is 0.0 for full agreement, up to 0.3 for disagreement
    """
    dsp_level = dsp.loudness_bucket
    llm_level = llm_result.energy_loudness.level.lower().strip()

    dsp_idx = _LOUDNESS_ORDER.get(dsp_level, 2)
    llm_idx = _LOUDNESS_ORDER.get(llm_level, 2)

    diff = abs(dsp_idx - llm_idx)

    if diff <= 1:
        return True, 0.0  # close enough
    elif diff == 2:
        return False, 0.15
    else:
        return False, 0.3


def _reconcile_abuse(
    transcript: str,
    llm_result: SpeakerResult,
) -> tuple[bool, float]:
    """Lexicon cross-check: if lexicon finds profanity and LLM didn't flag, escalate.

    Returns:
        (needs_human_review: bool, penalty: float)
    """
    lex_check = check_transcript(transcript)
    lexicon_found = lex_check["profanity_detected"]
    llm_flagged = llm_result.abuse.flagged

    if lexicon_found and not llm_flagged:
        logger.warning(
            "ABUSE MISMATCH: lexicon found profanity (%s) but LLM did not flag",
            lex_check.get("matches", []),
        )
        return True, 0.3

    if not lexicon_found and llm_flagged:
        logger.info(
            "LLM flagged abuse but lexicon found no profanity — may be nuance "
            "(threat/harassment without explicit profanity)"
        )
        return False, 0.0

    # agreement
    return False, 0.0


def _reconcile_arousal(dsp: DSPResult, llm_result: SpeakerResult) -> float:
    """F0 arousal signal discriminates emotion reads.

    If DSP shows high arousal, a "neutral/sad" emotion from LLM might be
    suspicious. Conversely, low arousal is inconsistent with anger/excitement.
    """
    if dsp.f0_variance is None:
        return 0.0

    arousal = dsp.high_arousal
    primary = llm_result.emotion.primary.lower()

    high_arousal_emotions = {"angry", "excited", "fear", "joyful", "surprised", "energetic", "anxious"}
    low_arousal_emotions = {"sad", "bored", "tired", "neutral", "calm", "depressed", "relaxed"}

    if arousal and primary in low_arousal_emotions:
        return 0.1  # slight penalty — arousal mismatch
    if not arousal and primary in high_arousal_emotions:
        return 0.1  # slight penalty — low arousal + high-arousal emotion

    return 0.0


def guardrail(
    llm_result: SpeakerResult,
    dsp_result: DSPResult,
) -> SpeakerResult:
    # 1. Loudness reconciliation
    dsp_agrees, loudness_penalty = _reconcile_energy_loudness(
        dsp_result, llm_result
    )
    llm_result.energy_loudness.dsp_agrees = dsp_agrees

    # 2. Abuse lexicon check
    abuse_conflict, abuse_penalty = _reconcile_abuse(
        llm_result.transcript, llm_result
    )

    # 3. Emotion arousal check
    arousal_penalty = _reconcile_arousal(dsp_result, llm_result)

    # 4. Aggregate confidence
    llm_conf = llm_result.confidence.overall if llm_result.confidence else 0.7
    total_penalty = loudness_penalty + abuse_penalty + arousal_penalty

    new_conf = max(0.0, min(1.0, llm_conf - total_penalty))
    needs_review = (
        not dsp_agrees
        or abuse_conflict
        or new_conf < 0.5
    )

    llm_result.confidence.overall = round(new_conf, 2)
    llm_result.confidence.needs_human_review = needs_review

    logger.info(
        "Person=%s (speaker=%s), DSP=%s, overall=%.2f, need review=%s, "
        "penalties: loudness=%.2f, abuse=%.2f, arousal=%.2f"
        ,
        llm_result.speaker_id,
        dsp_result.loudness_bucket,
        dsp_agrees,
        new_conf,
        needs_review,
        loudness_penalty,
        abuse_penalty,
        arousal_penalty,
    )

    return llm_result