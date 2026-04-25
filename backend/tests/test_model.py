"""Tests for the ML model and explainer."""

import pytest
from pathlib import Path

from backend.ml.framework import FRAMEWORK
from backend.ml.explainer import MaturityExplainer

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "ml" / "artifacts"


@pytest.fixture(scope="module")
def explainer():
    if not ARTIFACTS_DIR.exists():
        pytest.skip("Model artifacts not found; run training first.")
    return MaturityExplainer(ARTIFACTS_DIR)


@pytest.fixture
def sample_answers():
    return {
        "d1_q1": 4, "d1_q2": 3, "d1_q3": 4, "d1_q4": 4, "d1_q5": 3,
        "d2_q1": 2, "d2_q2": 3, "d2_q3": 2, "d2_q4": 3, "d2_q5": 3,
        "d3_q1": 4, "d3_q2": 3, "d3_q3": 4, "d3_q4": 4, "d3_q5": 4,
        "d4_q1": 4, "d4_q2": 3, "d4_q3": 3, "d4_q4": 3, "d4_q5": 3,
        "d5_q1": 3, "d5_q2": 3, "d5_q3": 3, "d5_q4": 2, "d5_q5": 2,
    }


def test_predict_returns_score_in_range(explainer, sample_answers):
    score = explainer.predict(sample_answers)
    assert 1.0 <= score <= 5.0


def test_predict_monotonic_high_inputs_higher_score(explainer):
    low = {q.id: 1 for q in [s.question for d in FRAMEWORK for s in d.sub_dimensions]}
    high = {q.id: 5 for q in [s.question for d in FRAMEWORK for s in d.sub_dimensions]}
    assert explainer.predict(high) > explainer.predict(low)


def test_predict_perfect_inputs_near_max(explainer):
    high = {q.id: 5 for q in [s.question for d in FRAMEWORK for s in d.sub_dimensions]}
    assert explainer.predict(high) > 4.5


def test_predict_minimum_inputs_near_min(explainer):
    low = {q.id: 1 for q in [s.question for d in FRAMEWORK for s in d.sub_dimensions]}
    # Model rarely sees pure 1s in training data, so allow some upward bias;
    # still must be far below the dataset mean (~3.1).
    assert explainer.predict(low) < 2.2


def test_explain_structure(explainer, sample_answers):
    exp = explainer.explain(sample_answers)
    assert "predicted_score" in exp
    assert "dimension_contributions" in exp
    assert "top_strengths" in exp
    assert "top_weaknesses" in exp
    assert len(exp["dimension_contributions"]) == 5


def test_what_if_improvement_increases_score(explainer, sample_answers):
    # Talent is weak in sample; improving it by +1 should raise the score
    result = explainer.what_if(sample_answers, "talent", delta=1.0)
    assert result.delta > 0


def test_what_if_degradation_decreases_score(explainer, sample_answers):
    result = explainer.what_if(sample_answers, "talent", delta=-1.0)
    assert result.delta < 0


def test_what_if_unknown_dimension_raises(explainer, sample_answers):
    with pytest.raises(ValueError):
        explainer.what_if(sample_answers, "nonexistent", delta=1.0)


def test_all_what_ifs_returns_five_results(explainer, sample_answers):
    results = explainer.all_what_ifs(sample_answers, delta=1.0)
    assert len(results) == 5
    # Sorted by impact descending
    deltas = [r["delta"] for r in results]
    assert deltas == sorted(deltas, reverse=True)
