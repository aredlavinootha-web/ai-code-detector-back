"""
main.py — FastAPI backend for CodeSentinel AI code detection.

Endpoints:
  GET  /api/health   — readiness probe (are models loaded?)
  POST /api/analyze  — run Tier 2 (statistical) + Tier 3 (ML) analysis
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from detector import load_models, models_ready, analyze as ml_analyze
from tier2 import load_resources, resources_ready, run_tier2, score_tier2

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading Tier 2 resources (word frequencies, NLTK data) …")
    load_resources()
    log.info("Tier 2 resources ready.")

    log.info("Downloading / loading ML models (first run may take a few minutes) …")
    load_models()
    log.info("Models ready — server accepting requests.")
    yield


app = FastAPI(
    title="CodeSentinel API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=100_000)
    language: str = Field(default="Python")


class ModelResult(BaseModel):
    score: int
    confidence: int
    label: str
    model: str
    chunks: int


class Tier2Result(BaseModel):
    metrics: dict
    score: int


class Tier3Result(BaseModel):
    score: int
    confidence: int
    models: dict[str, ModelResult]
    chunks_analyzed: int
    processing_time_ms: int


class AnalyzeResponse(BaseModel):
    tier2: Optional[Tier2Result] = None
    tier3: Optional[Tier3Result] = None


@app.get("/api/health")
def health():
    return {
        "status": "ok" if models_ready() and resources_ready() else "loading",
        "models_ready": models_ready(),
        "tier2_ready": resources_ready(),
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_code(req: AnalyzeRequest):
    result: dict = {}

    # Tier 2: Statistical analysis
    if resources_ready():
        try:
            t2_metrics = run_tier2(req.code, req.language)
            t2_score = score_tier2(t2_metrics)
            result["tier2"] = {"metrics": t2_metrics, "score": t2_score}
        except Exception:
            log.exception("Tier 2 analysis failed")

    # Tier 3: ML analysis
    if models_ready():
        try:
            ml_result = ml_analyze(req.code, req.language)
            result["tier3"] = ml_result
        except Exception:
            log.exception("Tier 3 analysis failed")

    if not result:
        raise HTTPException(503, "Backend resources are still loading. Try again shortly.")

    return result
