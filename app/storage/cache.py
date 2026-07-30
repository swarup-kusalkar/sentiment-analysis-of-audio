"""Redis caching for audio hash and diarization results.

Avoids re-processing identical audio files and re-running diarization on known content (Phase 8).
"""