"""Speaker diarization via pyannote.audio (Phase 3).

- Identify who spoke when (speaker_id, start_s, end_s)
- Returns RTTM-style timeline
- Results cached in Redis by audio hash
"""