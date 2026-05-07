"""
FastAPI Application — AI Maturity Assessor
==========================================
REST API exposing the framework, model and roadmap generation.

Endpoints:
- GET  /api/health
- GET  /api/framework            -> dimensions + questions
- POST /api/assess               -> score + per-dimension breakdown
- POST /api/explain              -> SHAP explanation
- POST /api/whatif               -> what-if simulation for one dimension
- POST /api/whatif/all           -> what-if for all dimensions
- POST /api/roadmap              -> personalized roadmap
- GET  /api/benchmark/{sector}   -> sector benchmark
- POST /api/full_report          -> end-to-end (assess + explain + roadmap)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.framework import (
    FRAMEWORK,
    classify_maturity,
    maturity_description,
    get_question_to_dimension_map,
)
from ml.explainer import MaturityExplainer
from llm.roadmap import generate_roadmap, generate_health_summary


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "ml" / "artifacts"
BENCHMARKS_PATH = PROJECT_ROOT / "data" / "synthetic" / "sector_benchmarks.json"

app = FastAPI(
    title="AI Maturity Assessor API",
    description=(
        "REST API for the AI Maturity Assessor — a tool that measures how ready "
        "an organization is to adopt AI, based on Gartner / MIT Sloan / Deloitte "
        "frameworks and a calibrated ML model with SHAP explainability."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lazy-loaded singletons
_explainer: Optional[MaturityExplainer] = None
_benchmarks: Optional[Dict[str, Any]] = None


def get_explainer() -> MaturityExplainer:
    global _explainer
    if _explainer is None:
        if not ARTIFACTS_DIR.exists():
            raise HTTPException(
                status_code=503,
                detail="Model artifacts not found. Run training first: "
                       "`python backend/ml/train.py`",
            )
        _explainer = MaturityExplainer(ARTIFACTS_DIR)
    return _explainer


def get_benchmarks() -> Dict[str, Any]:
    global _benchmarks
    if _benchmarks is None:
        if BENCHMARKS_PATH.exists():
            with open(BENCHMARKS_PATH) as f:
                _benchmarks = json.load(f)
        else:
            _benchmarks = {}
    return _benchmarks


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class AnswersRequest(BaseModel):
    """Raw answers: {question_id: 1..5}"""
    answers: Dict[str, int] = Field(..., description="Map of question_id -> 1..5")
    sector: Optional[str] = Field(None, description="Sector for benchmarking")
    company_size: Optional[str] = Field(None, description="startup|small|medium|large|enterprise")

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, v: Dict[str, int]) -> Dict[str, int]:
        all_qids = {s.question.id for d in FRAMEWORK for s in d.sub_dimensions}
        provided = set(v.keys())
        missing = all_qids - provided
        if missing:
            raise ValueError(f"Missing answers for: {sorted(missing)}")
        unknown = provided - all_qids
        if unknown:
            raise ValueError(f"Unknown question ids: {sorted(unknown)}")
        for qid, val in v.items():
            if not isinstance(val, int) or val < 1 or val > 5:
                raise ValueError(f"Answer for {qid} must be int in [1,5], got {val}")
        return v


class WhatIfRequest(AnswersRequest):
    dimension_id: str
    delta: float = Field(1.0, ge=-4.0, le=4.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_dimension_scores(answers: Dict[str, int]) -> Dict[str, float]:
    out = {}
    for dim in FRAMEWORK:
        ids = [s.question.id for s in dim.sub_dimensions]
        out[dim.id] = float(sum(answers[i] for i in ids) / len(ids))
    return out


def _benchmark_for_sector(sector: Optional[str], score: float) -> Optional[Dict]:
    if not sector:
        return None
    bms = get_benchmarks()
    if sector not in bms:
        return None
    bm = bms[sector]
    # Compute approximate percentile (using simple z-score on normal)
    from math import erf, sqrt
    mu, sigma = bm["mean_score"], bm["std_score"] or 1e-9
    z = (score - mu) / sigma
    percentile = 0.5 * (1 + erf(z / sqrt(2))) * 100
    return {
        "sector": sector,
        "n": bm["n"],
        "sector_mean": bm["mean_score"],
        "sector_std": bm["std_score"],
        "your_percentile": round(percentile, 1),
        "p25": bm["p25"],
        "p50": bm["p50"],
        "p75": bm["p75"],
        "dim_means": bm["dim_means"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ARTIFACTS_DIR.exists(),
        "benchmarks_loaded": BENCHMARKS_PATH.exists(),
    }


@app.get("/api/framework")
def get_framework():
    """Return the full framework: dimensions, sub-dimensions and questions."""
    return {
        "dimensions": [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "weight": d.weight,
                "sub_dimensions": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "description": s.description,
                        "question": {
                            "id": s.question.id,
                            "text": s.question.text,
                            "helper": s.question.helper,
                            "scale_labels": s.question.scale_labels,
                        },
                    }
                    for s in d.sub_dimensions
                ],
            }
            for d in FRAMEWORK
        ]
    }


@app.post("/api/assess")
def assess(req: AnswersRequest):
    """Score an assessment. Returns model prediction + per-dimension breakdown."""
    explainer = get_explainer()
    predicted = explainer.predict(req.answers)
    dim_scores = _compute_dimension_scores(req.answers)
    level = classify_maturity(predicted)
    benchmark = _benchmark_for_sector(req.sector, predicted)

    return {
        "predicted_score": round(predicted, 3),
        "maturity_level": level.value,
        "level_description": maturity_description(level),
        "dimension_scores": {k: round(v, 3) for k, v in dim_scores.items()},
        "benchmark": benchmark,
    }


@app.post("/api/explain")
def explain(req: AnswersRequest):
    """SHAP explanation of the predicted score."""
    explainer = get_explainer()
    return explainer.explain(req.answers, top_n_features=10)


@app.post("/api/whatif")
def whatif(req: WhatIfRequest):
    """Simulate improving (or degrading) one dimension by `delta`."""
    explainer = get_explainer()
    try:
        result = explainer.what_if(req.answers, req.dimension_id, req.delta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "dimension_id": result.dimension_id,
        "dimension_name": result.dimension_name,
        "current_avg": round(result.current_avg, 3),
        "new_avg": round(result.new_avg, 3),
        "current_score": round(result.current_score, 3),
        "new_score": round(result.new_score, 3),
        "delta": round(result.delta, 3),
    }


@app.post("/api/whatif/all")
def whatif_all(req: AnswersRequest):
    """What-if for all dimensions, +1 on each."""
    explainer = get_explainer()
    return {"results": explainer.all_what_ifs(req.answers, delta=1.0)}


@app.post("/api/roadmap")
def roadmap(req: AnswersRequest):
    """Generate a phased roadmap based on the assessment."""
    explainer = get_explainer()
    predicted = explainer.predict(req.answers)
    dim_scores = _compute_dimension_scores(req.answers)
    rm = generate_roadmap(
        dimension_scores=dim_scores,
        overall_score=predicted,
        sector=req.sector,
        company_size=req.company_size,
    )
    return rm


@app.get("/api/benchmark/{sector}")
def get_sector_benchmark(sector: str):
    bms = get_benchmarks()
    if sector not in bms:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown sector '{sector}'. Available: {list(bms.keys())}",
        )
    return {"sector": sector, **bms[sector]}


@app.post("/api/health_summary")
def health_summary(req: AnswersRequest):
    """Single-call AI health summary: overall status, per-dimension indicators, alerts and top action."""
    explainer = get_explainer()
    predicted = explainer.predict(req.answers)
    dim_scores = _compute_dimension_scores(req.answers)
    benchmark = _benchmark_for_sector(req.sector, predicted)
    what_ifs = explainer.all_what_ifs(req.answers, delta=1.0)
    return generate_health_summary(
        dimension_scores=dim_scores,
        overall_score=predicted,
        what_if_results=what_ifs,
        benchmark=benchmark,
        sector=req.sector,
    )


@app.post("/api/full_report")
def full_report(req: AnswersRequest):
    """Single-call report: assess + explain + roadmap."""
    explainer = get_explainer()
    predicted = explainer.predict(req.answers)
    dim_scores = _compute_dimension_scores(req.answers)
    level = classify_maturity(predicted)
    benchmark = _benchmark_for_sector(req.sector, predicted)
    explanation = explainer.explain(req.answers, top_n_features=10)
    whatifs = explainer.all_what_ifs(req.answers, delta=1.0)
    roadmap = generate_roadmap(
        dimension_scores=dim_scores,
        overall_score=predicted,
        sector=req.sector,
        company_size=req.company_size,
    )
    return {
        "assessment": {
            "predicted_score": round(predicted, 3),
            "maturity_level": level.value,
            "level_description": maturity_description(level),
            "dimension_scores": {k: round(v, 3) for k, v in dim_scores.items()},
            "benchmark": benchmark,
        },
        "explanation": explanation,
        "what_if_simulations": whatifs,
        "roadmap": roadmap,
    }


# Convenience for `python -m backend.api.app`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
