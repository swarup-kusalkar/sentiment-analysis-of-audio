import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.pipeline.ingest import ingest, IngestError
from app.pipeline.orchestrator import process_audio
from app.storage.cache import get_analysis


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyse", tags=["analysis"])


@router.post("/")
async def analyse(file: UploadFile = File(...)):
    # 1. Ingest
    try:
        audio = await ingest(file)
    except IngestError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # 2. Check Redis cache for full analysis
    cached = await get_analysis(audio.content_hash)
    if cached:
        logger.info("Cache hit for analysis: %s", audio.content_hash[:12])
        return JSONResponse(content=cached)

    # 3. Full pipeline
    try:
        response = await process_audio(audio)
    except Exception as exc:
        logger.exception("Pipeline failed for %s", audio.content_hash[:12])
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return JSONResponse(content=response.model_dump())