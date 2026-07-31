"""Lexicon-based profanity cross-check using better-profanity.

Used in the guardrail layer (Phase 7) to independently verify the LLM's abuse detection.
"""

import logging

logger = logging.getLogger(__name__)


def check_transcript(transcript: str) -> dict:
    try:
        from better_profanity import profanity
    except ImportError:
        logger.warning("better-profanity not installed; skipping lexicon check")
        return {"profanity_detected": False, "matches": []}

    words = transcript.split()
    profane = [w for w in words if profanity.contains_profanity(w)]

    return {
        "profanity_detected": len(profane) > 0,
        "matches": profane,
    }