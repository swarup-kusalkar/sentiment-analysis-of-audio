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
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings
from app.schemas.output import SpeakerResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "system_prompt.txt"
_FEW_SHOT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "few_shot_examples.json"


class LLMError(Exception):
    """Raised when the LLM call fails permanently after retries."""


def _load_prompts() -> tuple[str, list[dict]]:
    system_prompt = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    few_shot = json.loads(_FEW_SHOT_PATH.read_text(encoding="utf-8"))
    return system_prompt, few_shot


def _wav_to_data_uri(wav_path: str) -> str:
    raw = Path(wav_path).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _build_messages(
    system_prompt: str,
    few_shot_examples: list[dict],
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]

    for example in few_shot_examples:
        transcript = example["input_transcript"]
        output = example["expected_output"]
        messages.append({
            "role": "user",
            "content": (
                f"Analyse this speaker's audio. The transcript is provided as context: "
                f"\"{transcript}\"\n\n"
                f"Return the analysis as a JSON object matching the schema."
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
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    import re
    m = re.search(r"\{.*", text, re.DOTALL)
    if m:
        return m.group(0)
    return text


def _validate_to_schema(raw_json: str, speaker_id: str, start_s: float, end_s: float) -> SpeakerResult:
    data = json.loads(raw_json)

    data.setdefault("speaker_id", speaker_id)
    data.setdefault("segment_time_range", {"start_s": start_s, "end_s": end_s})

    emo = data.setdefault("emotion", {})
    emo.setdefault("primary", "unknown")
    emo.setdefault("secondary", [])
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
    energy.setdefault("level", "normal")
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
    client = OpenAI(
        base_url=settings.nvidia_base_url + "/v1",
        api_key=settings.nvidia_api_key,
        timeout=settings.llm_timeout_seconds,
    )

    response = client.chat.completions.create(
        model=settings.nvidia_model,
        messages=messages,
        temperature=settings.llm_temperature,
        top_p=1,
        max_tokens=4096,
        stream=False,
        extra_body={
            "reasoning_budget": settings.llm_reasoning_budget,
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from NVIDIA API")

    tokens = response.usage.total_tokens
    reasoning_used = getattr(response.usage, "reasoning_tokens", 0)
    logger.info(
        "LLM call: %s total tokens (%s reasoning tokens)",
        tokens,
        reasoning_used or "N/A",
    )

    return content.strip()


def analyse_chunk(
    wav_path: str,
    speaker_id: str,
    start_s: float,
    end_s: float,
) -> SpeakerResult:
    audio_uri = _wav_to_data_uri(wav_path)
    system_prompt, few_shot = _load_prompts()
    messages = _build_messages(system_prompt, few_shot)

    # Build the final user message with audio and schema instruction
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
            {
                "type": "input_audio",
                "input_audio": {"data": audio_uri},
            },
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
        "Chunk %s analysed: emotion=%s, approved=%s",
        speaker_id,
        result.emotion.primary,
        result.abuse.flagged,
    )
    return result