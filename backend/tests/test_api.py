"""Tests for the FastAPI endpoints."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.ml.framework import FRAMEWORK

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "ml" / "artifacts"


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_payload():
    answers = {
        "d1_q1": 4, "d1_q2": 3, "d1_q3": 4, "d1_q4": 4, "d1_q5": 3,
        "d2_q1": 2, "d2_q2": 3, "d2_q3": 2, "d2_q4": 3, "d2_q5": 3,
        "d3_q1": 4, "d3_q2": 3, "d3_q3": 4, "d3_q4": 4, "d3_q5": 4,
        "d4_q1": 4, "d4_q2": 3, "d4_q3": 3, "d4_q4": 3, "d4_q5": 3,
        "d5_q1": 3, "d5_q2": 3, "d5_q3": 3, "d5_q4": 2, "d5_q5": 2,
    }
    return {"answers": answers, "sector": "manufacturing", "company_size": "medium"}


def _skip_if_no_artifacts():
    if not ARTIFACTS_DIR.exists():
        pytest.skip("Model artifacts not found; run training first.")


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_framework_endpoint(client):
    r = client.get("/api/framework")
    assert r.status_code == 200
    data = r.json()
    assert len(data["dimensions"]) == 5
    total_subs = sum(len(d["sub_dimensions"]) for d in data["dimensions"])
    assert total_subs == 25


def test_assess(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/assess", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert 1.0 <= body["predicted_score"] <= 5.0
    assert "maturity_level" in body
    assert "benchmark" in body
    assert body["benchmark"]["sector"] == "manufacturing"


def test_assess_missing_answer_returns_422(client):
    _skip_if_no_artifacts()
    r = client.post("/api/assess", json={"answers": {"d1_q1": 3}})
    assert r.status_code == 422


def test_assess_out_of_range_returns_422(client, sample_payload):
    _skip_if_no_artifacts()
    bad = sample_payload.copy()
    bad["answers"] = sample_payload["answers"].copy()
    bad["answers"]["d1_q1"] = 9  # invalid
    r = client.post("/api/assess", json=bad)
    assert r.status_code == 422


def test_explain(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/explain", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert "top_features" in body
    assert "dimension_contributions" in body
    assert len(body["dimension_contributions"]) == 5


def test_whatif(client, sample_payload):
    _skip_if_no_artifacts()
    payload = {**sample_payload, "dimension_id": "talent", "delta": 1.0}
    r = client.post("/api/whatif", json=payload)
    assert r.status_code == 200
    assert r.json()["delta"] > 0


def test_whatif_unknown_dim(client, sample_payload):
    _skip_if_no_artifacts()
    payload = {**sample_payload, "dimension_id": "unknown", "delta": 1.0}
    r = client.post("/api/whatif", json=payload)
    assert r.status_code == 400


def test_whatif_all(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/whatif/all", json=sample_payload)
    assert r.status_code == 200
    assert len(r.json()["results"]) == 5


def test_roadmap(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/roadmap", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["phases"]) == 3
    # At least one initiative across all phases
    total = sum(len(p["initiatives"]) for p in body["phases"])
    assert total > 0


def test_full_report(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/full_report", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"assessment", "explanation", "what_if_simulations", "roadmap"}


def test_benchmark_endpoint(client):
    _skip_if_no_artifacts()
    r = client.get("/api/benchmark/manufacturing")
    if r.status_code == 200:
        body = r.json()
        assert body["sector"] == "manufacturing"
        assert "mean_score" in body
    else:
        # No benchmarks generated yet
        assert r.status_code in (404, 503)


def test_benchmark_unknown_sector(client):
    r = client.get("/api/benchmark/__unknown__")
    assert r.status_code == 404


def test_health_summary_structure(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/health_summary", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    required_keys = {
        "generated_at", "overall_score", "maturity_level",
        "health_status", "health_color", "summary_headline",
        "dimension_health", "alerts", "top_action",
        "improvement_potential", "benchmark_percentile", "sector",
    }
    assert required_keys <= set(body.keys())


def test_health_summary_dimension_health(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/health_summary", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert len(body["dimension_health"]) == 5
    for d in body["dimension_health"]:
        assert d["status"] in ("healthy", "fair", "at_risk", "critical")
        assert 1.0 <= d["score"] <= 5.0


def test_health_summary_score_range(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/health_summary", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert 1.0 <= body["overall_score"] <= 5.0
    assert body["health_status"] in ("excellent", "good", "fair", "poor", "critical")


def test_health_summary_top_action(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/health_summary", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert body["top_action"]["dimension_id"] in ("data", "talent", "technology", "strategy", "processes")
    assert isinstance(body["top_action"]["title"], str)
    assert len(body["top_action"]["title"]) > 0


def test_health_summary_improvement_potential_non_negative(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/health_summary", json=sample_payload)
    assert r.status_code == 200
    assert r.json()["improvement_potential"] >= 0.0


def test_health_summary_with_sector_sets_percentile(client, sample_payload):
    _skip_if_no_artifacts()
    r = client.post("/api/health_summary", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    if body["benchmark_percentile"] is not None:
        assert 0.0 <= body["benchmark_percentile"] <= 100.0


def test_health_summary_missing_answers_returns_422(client):
    _skip_if_no_artifacts()
    r = client.post("/api/health_summary", json={"answers": {"d1_q1": 3}})
    assert r.status_code == 422
