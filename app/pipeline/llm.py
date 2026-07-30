"""Nemotron Omni API integration (Phase 6).

- Sends speaker-segment WAV audio to NVIDIA API
- Constructs system prompt with strict JSON schema
- Includes few-shot examples for stream output
- Parses LLM JSON response into SpeakerResult
- Handles basic retries on malformed JSON
"""