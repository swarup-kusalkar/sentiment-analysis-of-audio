"""Audio ingestion and normalisation (Phase 1).

- Save uploaded audio to temp directory
- Convert to 16 kHz mono 16-bit WAV via ffmpeg
- Validate minimum duration
- Generate SHA-256 content hash
"""

import hashlib
import json
import logging
import os
import subprocess
import uuid
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class IngestError(Exception):
    """Raised when ingestion fails for any reason."""


class AudioFile:
    """Represents a validated, normalised audio file ready for the pipeline."""

    def __init__(
        self,
        original_name: str,
        normalised_path: str,
        content_hash: str,
        duration_s: float,
        sample_rate: int,
        channels: int,
    ):
        self.original_name = original_name
        self.normalised_path = normalised_path
        self.content_hash = content_hash
        self.duration_s = duration_s
        self.sample_rate = sample_rate
        self.channels = channels

    def __repr__(self) -> str:
        return (
            f"AudioFile(original={self.original_name!r}, "
            f"duration={self.duration_s:.1f}s, hash={self.content_hash[:12]}…)"
        )


def _ensure_temp_dir() -> None:
    os.makedirs(settings.temp_dir, exist_ok=True)


def _get_probe(path: str) -> dict:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        result.check_returncode()
    except subprocess.CalledProcessError as exc:
        raise IngestError(f"ffprobe failed on {path}: {exc.stderr.strip()}") from exc
    except FileNotFoundError:
        raise IngestError("ffprobe not found — is ffmpeg installed?")

    probe = json.loads(result.stdout)

    audio_streams = [s for s in probe.get("streams", []) if s["codec_type"] == "audio"]
    if not audio_streams:
        raise IngestError("No audio stream found in file")

    fmt = probe.get("format", {})
    return {
        "duration": float(fmt.get("duration", 0)),
        "sample_rate": int(audio_streams[0].get("sample_rate", 0)),
        "channels": int(audio_streams[0].get("channels", 0)),
        "codec_name": audio_streams[0].get("codec_name", "unknown"),
    }


async def _save_upload(upload: "UploadFile") -> str:
    """Save the uploaded file to the temp directory and return the raw path."""
    import aiofiles

    _ensure_temp_dir()
    ext = Path(upload.filename or "upload").suffix or ".bin"
    raw_name = f"{uuid.uuid4().hex}_raw{ext}"
    raw_path = os.path.join(settings.temp_dir, raw_name)

    try:
        async with aiofiles.open(raw_path, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                await f.write(chunk)
    except Exception as exc:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        raise IngestError(f"Failed to save upload: {exc}") from exc

    file_size_mb = os.path.getsize(raw_path) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        os.remove(raw_path)
        raise IngestError(
            f"File too large ({file_size_mb:.1f} MB). Max is {settings.max_file_size_mb} MB."
        )

    logger.info("Saved raw upload: %s (%.1f MB)", raw_path, file_size_mb)
    return raw_path


def _normalise(raw_path: str) -> str:
    """Convert raw audio to 16kHz mono 16-bit PCM WAV via ffmpeg.

    Returns path to the normalised file.
    """
    normalised_name = f"{uuid.uuid4().hex}.wav"
    normalised_path = os.path.join(settings.temp_dir, normalised_name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", raw_path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "wav",
        normalised_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        result.check_returncode()
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg failed: %s", exc.stderr.strip())
        raise IngestError(f"Audio conversion failed: {exc.stderr.strip()}") from exc
    except FileNotFoundError:
        raise IngestError("ffmpeg not found — is it installed?")

    if not os.path.exists(normalised_path) or os.path.getsize(normalised_path) == 0:
        raise IngestError("Normalised file is empty — conversion produced no output")

    logger.info("Normalised audio saved: %s", normalised_path)
    return normalised_path


def _compute_content_hash(normalised_path: str) -> str:
    """Compute SHA-256 hash of the normalised WAV (binary read)."""
    sha = hashlib.sha256()
    with open(normalised_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    logger.info("Content hash: %s", digest)
    return digest


async def ingest(upload: "UploadFile") -> AudioFile:
    """Full ingestion pipeline: save → normalise → probe → hash → return AudioFile.

    Raises HTTPException (4xx) for client errors. Raises IngestError for
    system-level failures (ffmpeg missing, disk full, etc.).
    """
    from fastapi import HTTPException

    if not upload.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    _ensure_temp_dir()
    raw_path = await _save_upload(upload)
    normalised_path = _normalise(raw_path)
    probe = _get_probe(normalised_path)

    duration = probe["duration"]
    if duration < settings.min_duration_seconds:
        os.remove(normalised_path)
        raise HTTPException(
            status_code=400,
            detail=f"Audio too short ({duration:.1f}s < {settings.min_duration_seconds}s).",
        )

    content_hash = _compute_content_hash(normalised_path)

    try:
        os.remove(raw_path)
    except OSError:
        pass

    audio_file = AudioFile(
        original_name=upload.filename,
        normalised_path=normalised_path,
        content_hash=content_hash,
        duration_s=duration,
        sample_rate=probe["sample_rate"],
        channels=probe["channels"],
    )

    logger.info(
        "Ingest complete: %s → %s, %.1fs, %s",
        upload.filename,
        normalised_path,
        duration,
        content_hash,
    )
    return audio_file