"""
FastAPI REST API for document classification inference.
src/api/app.py
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

from src.utils.config import load_config


app = FastAPI(
    title="NLP Document Classification API",
    description=(
        "Classify documents into categories using trained NLP models. "
        "Supports single-document and batch classification with confidence scores."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClassifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=10000,
        description="The document text to classify.",
        example=(
            "Scientists at MIT discovered a new exoplanet in the habitable zone, "
            "potentially capable of supporting liquid water."
        ),
    )
    model: Optional[str] = Field(
        default="logistic_regression",
        description="Model to use: logistic_regression | svm | naive_bayes | random_forest",
    )
    top_k: Optional[int] = Field(
        default=3,
        ge=1,
        le=8,
        description="Number of top predictions to return.",
    )


class BatchClassifyRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of documents to classify (max 100).",
    )
    model: Optional[str] = Field(default="logistic_regression")


class PredictionResult(BaseModel):
    predicted_label: str
    confidence: float
    top_predictions: List[Dict[str, Any]]
    model_used: str
    inference_time_ms: float


class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    total_documents: int
    total_time_ms: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    available_categories: List[str]
    version: str



LOADED_MODELS: Dict[str, Any] = {}
CATEGORIES: List[str] = []
PREPROCESSOR = None


def _load_all_models():
    """Load all available trained models from artifacts/."""
    global CATEGORIES, PREPROCESSOR

    config = load_config("configs/config.yaml")
    artifacts_dir = Path(config.get("training", {}).get("artifacts_dir", "artifacts"))

    from src.data.preprocessor import TextPreprocessor
    from src.models.traditional import TraditionalClassifier

    PREPROCESSOR = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatization=True,
        min_word_length=2,
    )

    model_names = ["logistic_regression", "svm", "naive_bayes", "random_forest"]
    for name in model_names:
        model_path = artifacts_dir / "models" / name / f"{name}_pipeline.joblib"
        if model_path.exists():
            try:
                model = TraditionalClassifier.load(str(model_path))
                LOADED_MODELS[name] = model
                if not CATEGORIES and model.categories:
                    CATEGORIES = model.categories
                logger.success(f"Loaded model: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

    if not LOADED_MODELS:
        logger.warning(
            "No trained models found. Run 'python train.py' first, "
            "then restart the API server."
        )


@app.on_event("startup")
async def startup_event():
    logger.info("Starting NLP Classification API...")
    _load_all_models()
    logger.info(f"Loaded {len(LOADED_MODELS)} models: {list(LOADED_MODELS.keys())}")




@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    response.headers["X-Process-Time-MS"] = f"{elapsed:.2f}"
    return response




@app.get("/", tags=["root"])
async def root():
    return {
        "message": "NLP Document Classification API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health():
    return HealthResponse(
        status="healthy" if LOADED_MODELS else "no_models_loaded",
        loaded_models=list(LOADED_MODELS.keys()),
        available_categories=CATEGORIES,
        version="1.0.0",
    )


@app.get("/models", tags=["models"])
async def list_models():
    """List all available classification models."""
    return {
        "models": list(LOADED_MODELS.keys()),
        "default": "logistic_regression",
        "categories": CATEGORIES,
    }


@app.post("/classify", response_model=PredictionResult, tags=["classification"])
async def classify(request: ClassifyRequest):
    """
    Classify a single document.

    Returns the predicted category, confidence score, and top-K predictions.
    """
    model_key = request.model or "logistic_regression"

    if not LOADED_MODELS:
        raise HTTPException(
            status_code=503,
            detail="No models loaded. Run 'python train.py' first.",
        )
    if model_key not in LOADED_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not found. Available: {list(LOADED_MODELS.keys())}",
        )

    model = LOADED_MODELS[model_key]

    # Preprocess
    clean_text = PREPROCESSOR.preprocess(request.text) if PREPROCESSOR else request.text

    # Predict
    t0 = time.time()
    result = model.predict_single(clean_text)
    elapsed_ms = (time.time() - t0) * 1000

    # Build top-K predictions
    sorted_probs = sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    )
    top_k = min(request.top_k or 3, len(sorted_probs))
    top_predictions = [
        {"label": label, "confidence": conf}
        for label, conf in sorted_probs[:top_k]
    ]

    return PredictionResult(
        predicted_label=result["predicted_label"],
        confidence=result["confidence"],
        top_predictions=top_predictions,
        model_used=model_key,
        inference_time_ms=round(elapsed_ms, 3),
    )


@app.post("/classify/batch", response_model=BatchPredictionResult, tags=["classification"])
async def classify_batch(request: BatchClassifyRequest):
    """
    Classify a batch of documents (up to 100).

    Returns a prediction for each document.
    """
    model_key = request.model or "logistic_regression"

    if not LOADED_MODELS:
        raise HTTPException(status_code=503, detail="No models loaded.")
    if model_key not in LOADED_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_key}' not found. Available: {list(LOADED_MODELS.keys())}",
        )

    model = LOADED_MODELS[model_key]

    t0 = time.time()
    results = []
    for text in request.texts:
        clean = PREPROCESSOR.preprocess(text) if PREPROCESSOR else text
        result = model.predict_single(clean)
        sorted_probs = sorted(
            result["probabilities"].items(), key=lambda x: x[1], reverse=True
        )
        top_predictions = [
            {"label": label, "confidence": conf}
            for label, conf in sorted_probs[:3]
        ]
        results.append(
            PredictionResult(
                predicted_label=result["predicted_label"],
                confidence=result["confidence"],
                top_predictions=top_predictions,
                model_used=model_key,
                inference_time_ms=0.0,
            )
        )

    total_ms = (time.time() - t0) * 1000
    return BatchPredictionResult(
        predictions=results,
        total_documents=len(request.texts),
        total_time_ms=round(total_ms, 2),
        model_used=model_key,
    )


@app.get("/categories", tags=["models"])
async def get_categories():
    """Return all classification categories."""
    return {"categories": CATEGORIES, "count": len(CATEGORIES)}



if __name__ == "__main__":
    config = load_config("configs/config.yaml")
    api_cfg = config.get("api", {})
    uvicorn.run(
        "src.api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        reload=api_cfg.get("reload", False),
    )
