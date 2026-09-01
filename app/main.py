import os
from pathlib import Path

os.environ.setdefault("TORCHAUDIO_USE_SOUNDFILE", "1")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import certifi
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router
from app.api.routes.analysis import router as analysis_router

STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(
    title="Audio Sentiment & Paralinguistic Analysis",
    version="0.1.0",
    description="Analyse audio call recordings for sentiment, tone, abuse, and paralinguistic features per speaker.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(analysis_router)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
