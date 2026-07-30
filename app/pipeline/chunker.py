"""Speaker-boundary-aware audio chunking (Phase 4).

- Uses diarization timeline to slice audio
- Never crosses speaker boundaries
- Respects 60–120 s chunk duration limits
- Extracts per/chunk WAV files to temp dir
"""