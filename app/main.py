from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes.health import router as health_router
from app.api.routes.analysis import router as analysis_router

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

app.mount("/static", StaticFiles(directory="static"), name="static")