import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline.ingest import ingest, IngestError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyse", tags=["analysis"])


@router.post("/")
async def analyse(file: UploadFile = File(...)):
    try:
        audio = await ingest(file)
    except IngestError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(
        content={
            "status": "ingested",
            "original_name": audio.original_name,
            "content_hash": audio.content_hash,
            "duration_seconds": round(audio.duration_s, 2),
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "normalised_path": audio.normalised_path,
        }
    )