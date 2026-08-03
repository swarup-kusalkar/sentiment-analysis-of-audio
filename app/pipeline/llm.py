"""Nemotron Omni API integration (Phase 6).

- Sends speaker-segment WAV audio to NVIDIA API
- Constructs system prompt with strict JSON schema
- Includes four few-shot examples
- Parses LLM JSON response into SpeakerResult
- Handles retries on malformed JSON via tenacity
"""

import base64
import json
import logging
import re
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.schemas.output import SpeakerResult

logger = logging.getLogger(__name__)

_VALID_LOUDNESS = {"whisper", "quiet", "normal", "loud", "shouting"}
_PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
_client: OpenAI | None = None


class LLMError(Exception):
    pass


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
            timeout=settings.llm_timeout_seconds,
        )
    return _client


def _load_prompts() -> tuple[str, list[dict]]:
    system_prompt = (_PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8").strip()
    few_shot = json.loads((_PROMPT_DIR / "few_shot_examples.json").read_text(encoding="utf-8"))
    return system_prompt, few_shot


def _wav_to_base64(wav_path: str) -> str:
    return base64.b64encode(Path(wav_path).read_bytes()).decode("ascii")


def _build_messages(system_prompt: str, few_shot_examples: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for example in few_shot_examples:
        transcript = example["input_transcript"]
        output = example["expected_output"]
        messages.append({
            "role": "user",
            "content": (
                f"Analyse this speaker's audio. The transcript is provided as context: "
                f"\"{transcript}\"\n\nReturn the analysis as a JSON object matching the schema."
            ),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(output, ensure_ascii=False),
        })
    return messages


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*", text, re.DOTALL)
    return m.group(0) if m else text


def _normalize_loudness(level: str) -> str:
    level_lower = level.strip().lower()
    for valid in _VALID_LOUDNESS:
        if valid in level_lower:
            return valid
    return "normal"


def _validate_to_schema(raw_json: str, speaker_id: str, start_s: float, end_s: float) -> SpeakerResult:
    data = json.loads(raw_json)

    data.setdefault("speaker_id", speaker_id)
    data.setdefault("segment_time_range", {"start_s": start_s, "end_s": end_s})

    emo = data.setdefault("emotion", {})
    emo.setdefault("primary", "unknown")
    secondary = emo.setdefault("secondary", [])
    if not isinstance(secondary, list):
        emo["secondary"] = [secondary] if secondary else []
    emo.setdefault("reasoning", "")

    tone = data.setdefault("tone", {})
    tone.setdefault("label", "unknown")
    tone.setdefault("reasoning", "")

    abuse = data.setdefault("abuse", {})
    abuse.setdefault("flagged", False)
    abuse.setdefault("category", "none")
    abuse.setdefault("severity_1_to_5", 0)
    abuse.setdefault("evidence_span", None)
    abuse.setdefault("reasoning", "")

    energy = data.setdefault("energy_loudness", {})
    raw_level = energy.setdefault("level", "normal")
    energy["level"] = _normalize_loudness(raw_level)
    energy.setdefault("description", "")
    energy.setdefault("dsp_agrees", True)

    data.setdefault("summary", "")
    conf = data.setdefault("confidence", {})
    conf.setdefault("overall", 0.7)
    conf.setdefault("needs_human_review", False)

    return SpeakerResult(**data)


@retry(
    stop=stop_after_attempt(settings.llm_max_retries + 1),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_api(messages: list[dict]) -> str:
    response = _get_client().chat.completions.create(
        model=settings.nvidia_model,
        messages=messages,
        temperature=settings.llm_temperature,
        top_p=1,
        max_tokens=4096,
        stream=False,
        extra_body={"reasoning_budget": settings.llm_reasoning_budget},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from NVIDIA API")

    tokens = response.usage.total_tokens
    reasoning_used = getattr(response.usage, "reasoning_tokens", 0)
    logger.info("LLM call: %s total tokens (%s reasoning tokens)", tokens, reasoning_used or "N/A")

    return content.strip()


def analyse_chunk(wav_path: str, speaker_id: str, start_s: float, end_s: float) -> SpeakerResult:
    audio_b64 = _wav_to_base64(wav_path)
    system_prompt, few_shot = _load_prompts()
    messages = _build_messages(system_prompt, few_shot)

    schema_instruction = (
        "Analyse the attached audio and return ONLY a JSON object with these fields:\n"
        "transcript, emotion (primary, secondary, reasoning), tone (label, reasoning),\n"
        "abuse (flagged, category, severity_1_to_5, evidence_span, reasoning),\n"
        "energy_loudness (level, description, dsp_agrees), summary.\n"
        "No markdown, no code fences, no extra text."
    )

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": schema_instruction},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
        ],
    })

    try:
        raw_text = _call_api(messages)
    except Exception as exc:
        logger.error("LLM call failed after %d retries: %s", settings.llm_max_retries + 1, exc)
        raise LLMError(f"LLM API call failed: {exc}") from exc

    json_str = _extract_json(raw_text)

    try:
        result = _validate_to_schema(json_str, speaker_id, start_s, end_s)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.error("Failed to parse LLM response into SpeakerResult: %s", exc)
        logger.debug("Raw response: %s", raw_text[:500])
        raise LLMError(f"Decoding failed: {exc}")

    logger.info(
        "Chunk %s analysed: emotion=%s, abuse=%s",
        speaker_id,
        result.emotion.primary,
        result.abuse.flagged,
    )
    return result