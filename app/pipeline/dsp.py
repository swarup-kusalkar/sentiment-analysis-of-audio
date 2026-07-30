"""DSP grounding: energy/loudness + F0 variance (Phase 5).

- RMS energy (dBFS) → loudness bucket (whisper…shouting)
- F0 variance → soft arousal signal (no user facing output)
- Both computed in one audio pass via librosa + praat-parselmouth
"""