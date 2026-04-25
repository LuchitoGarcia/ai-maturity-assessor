"""
SHAP Explainability Module
==========================
Provides interpretability for individual predictions using SHAP values
and what-if sensitivity analysis.

Outputs:
- Per-feature SHAP contributions (which inputs raised/lowered the score)
- Per-dimension contributions (aggregated for business consumption)
- What-if simulations: "if you improve X by 1 point, score becomes Y"
- Top strengths and top weaknesses
"""

import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.framework import FRAMEWORK
from ml.train import engineer_features, get_feature_columns


# ---------------------------------------------------------------------------
# Data classes for outputs
# ---------------------------------------------------------------------------

@dataclass
class FeatureContribution:
    feature: str
    feature_label: str
    value: float
    shap_value: float
    contribution_pct: float


@dataclass
class DimensionContribution:
    dimension_id: str
    dimension_name: str
    avg_score: float
    shap_total: float
    contribution_pct: float


@dataclass
class WhatIfResult:
    dimension_id: str
    dimension_name: str
    current_avg: float
    new_avg: float
    current_score: float
    new_score: float
    delta: float


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class MaturityExplainer:
    """SHAP-based explainer for the maturity model."""

    def __init__(self, artifacts_dir: Path):
        artifacts_dir = Path(artifacts_dir)
        self.model = joblib.load(artifacts_dir / "xgboost_model.pkl")
        with open(artifacts_dir / "feature_columns.json") as f:
            self.feature_columns = json.load(f)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_labels = self._build_feature_labels()

    def _build_feature_labels(self) -> Dict[str, str]:
        """Human-friendly names for features."""
        labels = {}
        for dim in FRAMEWORK:
            for sub in dim.sub_dimensions:
                labels[sub.question.id] = f"{dim.name}: {sub.name}"
            labels[f"avg_{dim.id}"] = f"{dim.name} (overall)"
            labels[f"consistency_{dim.id}"] = f"{dim.name} consistency"
        labels.update({
            "gap_strategy_vs_tech": "Strategy vs Technology gap",
            "gap_data_vs_talent": "Data vs Talent gap",
            "gap_strategy_vs_processes": "Strategy vs Processes gap",
            "dim_variance": "Cross-dimension variance",
            "dim_min": "Weakest dimension score",
            "dim_max": "Strongest dimension score",
            "dim_range": "Range across dimensions",
            "foundation_score": "Foundation (Data/Tech/Talent)",
            "execution_score": "Execution (Strategy/Processes)",
            "quick_win_signal": "Quick-win opportunity signal",
        })
        return labels

    # ---- Prediction --------------------------------------------------------

    def _to_features(self, answers: Dict[str, int]) -> pd.DataFrame:
        """Convert raw answers (q_id -> 1..5) to engineered feature row."""
        df = pd.DataFrame([answers])
        df_feat = engineer_features(df)
        # Ensure column order matches training
        return df_feat[self.feature_columns]

    def predict(self, answers: Dict[str, int]) -> float:
        X = self._to_features(answers)
        return float(self.model.predict(X)[0])

    # ---- Explainability ----------------------------------------------------

    def explain(self, answers: Dict[str, int],
                top_n_features: int = 10) -> Dict:
        """
        Full explanation of a prediction:
        - score (predicted maturity)
        - per-feature SHAP contributions (top N)
        - per-dimension aggregated SHAP contributions
        - top strengths, top weaknesses
        """
        X = self._to_features(answers)
        prediction = float(self.model.predict(X)[0])
        shap_values = self.explainer.shap_values(X)[0]  # 1D array

        total_abs = np.abs(shap_values).sum()
        if total_abs == 0:
            total_abs = 1e-9

        # Per-feature contributions
        feature_contribs: List[FeatureContribution] = []
        for col, sv in zip(self.feature_columns, shap_values):
            label = self.feature_labels.get(col, col)
            feature_contribs.append(FeatureContribution(
                feature=col,
                feature_label=label,
                value=float(X.iloc[0][col]),
                shap_value=float(sv),
                contribution_pct=float(abs(sv) / total_abs * 100),
            ))

        # Sort by absolute contribution
        feature_contribs.sort(key=lambda x: abs(x.shap_value), reverse=True)
        top_features = feature_contribs[:top_n_features]

        # Per-dimension aggregated contribution
        dim_contribs = self._aggregate_by_dimension(shap_values, X.iloc[0])

        # Strengths and weaknesses (raw answers, not engineered)
        raw_only = [fc for fc in feature_contribs if fc.feature in
                    [s.question.id for d in FRAMEWORK for s in d.sub_dimensions]
                    or fc.feature.startswith("avg_")]
        positives = sorted([fc for fc in raw_only if fc.shap_value > 0],
                           key=lambda x: x.shap_value, reverse=True)[:3]
        negatives = sorted([fc for fc in raw_only if fc.shap_value < 0],
                           key=lambda x: x.shap_value)[:3]

        return {
            "predicted_score": prediction,
            "base_value": float(self.explainer.expected_value),
            "top_features": [self._fc_to_dict(fc) for fc in top_features],
            "dimension_contributions": [self._dc_to_dict(dc) for dc in dim_contribs],
            "top_strengths": [self._fc_to_dict(fc) for fc in positives],
            "top_weaknesses": [self._fc_to_dict(fc) for fc in negatives],
        }

    def _aggregate_by_dimension(self, shap_values: np.ndarray,
                                feature_row: pd.Series
                                ) -> List[DimensionContribution]:
        """Aggregate SHAP values by dimension (sum across all related features)."""
        feat_to_shap = dict(zip(self.feature_columns, shap_values))
        results = []
        all_abs = sum(abs(v) for v in shap_values)
        if all_abs == 0:
            all_abs = 1e-9

        for dim in FRAMEWORK:
            related_features = [s.question.id for s in dim.sub_dimensions]
            related_features += [f"avg_{dim.id}", f"consistency_{dim.id}"]
            shap_total = sum(feat_to_shap.get(f, 0.0) for f in related_features)
            avg_score = float(feature_row[f"avg_{dim.id}"])
            results.append(DimensionContribution(
                dimension_id=dim.id,
                dimension_name=dim.name,
                avg_score=avg_score,
                shap_total=float(shap_total),
                contribution_pct=float(abs(shap_total) / all_abs * 100),
            ))
        return results

    # ---- What-if simulations -----------------------------------------------

    def what_if(self, answers: Dict[str, int],
                dimension_id: str, delta: float) -> WhatIfResult:
        """
        Simulate improving (or degrading) all questions of a dimension by `delta`.
        Returns the new predicted score.
        """
        dim = next((d for d in FRAMEWORK if d.id == dimension_id), None)
        if dim is None:
            raise ValueError(f"Unknown dimension: {dimension_id}")

        current_score = self.predict(answers)

        modified = answers.copy()
        for sub in dim.sub_dimensions:
            qid = sub.question.id
            new_val = np.clip(modified[qid] + delta, 1, 5)
            modified[qid] = int(round(new_val))

        new_score = self.predict(modified)

        current_avg = np.mean([answers[s.question.id] for s in dim.sub_dimensions])
        new_avg = np.mean([modified[s.question.id] for s in dim.sub_dimensions])

        return WhatIfResult(
            dimension_id=dim.id,
            dimension_name=dim.name,
            current_avg=float(current_avg),
            new_avg=float(new_avg),
            current_score=float(current_score),
            new_score=float(new_score),
            delta=float(new_score - current_score),
        )

    def all_what_ifs(self, answers: Dict[str, int],
                     delta: float = 1.0) -> List[Dict]:
        """Run what-if for all dimensions."""
        results = []
        for dim in FRAMEWORK:
            wi = self.what_if(answers, dim.id, delta)
            results.append({
                "dimension_id": wi.dimension_id,
                "dimension_name": wi.dimension_name,
                "current_avg": wi.current_avg,
                "new_avg": wi.new_avg,
                "current_score": wi.current_score,
                "new_score": wi.new_score,
                "delta": wi.delta,
            })
        # Sorted by impact (descending)
        results.sort(key=lambda r: r["delta"], reverse=True)
        return results

    # ---- Helpers -----------------------------------------------------------

    @staticmethod
    def _fc_to_dict(fc: FeatureContribution) -> Dict:
        return {
            "feature": fc.feature,
            "feature_label": fc.feature_label,
            "value": fc.value,
            "shap_value": fc.shap_value,
            "contribution_pct": fc.contribution_pct,
        }

    @staticmethod
    def _dc_to_dict(dc: DimensionContribution) -> Dict:
        return {
            "dimension_id": dc.dimension_id,
            "dimension_name": dc.dimension_name,
            "avg_score": dc.avg_score,
            "shap_total": dc.shap_total,
            "contribution_pct": dc.contribution_pct,
        }


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    artifacts_dir = project_root / "backend" / "ml" / "artifacts"

    explainer = MaturityExplainer(artifacts_dir)

    # Sample company answers (fairly mature in tech, weak in talent)
    sample_answers = {
        "d1_q1": 4, "d1_q2": 3, "d1_q3": 4, "d1_q4": 4, "d1_q5": 3,  # data
        "d2_q1": 2, "d2_q2": 3, "d2_q3": 2, "d2_q4": 3, "d2_q5": 3,  # talent
        "d3_q1": 4, "d3_q2": 3, "d3_q3": 4, "d3_q4": 4, "d3_q5": 4,  # tech
        "d4_q1": 4, "d4_q2": 3, "d4_q3": 3, "d4_q4": 3, "d4_q5": 3,  # strategy
        "d5_q1": 3, "d5_q2": 3, "d5_q3": 3, "d5_q4": 2, "d5_q5": 2,  # processes
    }

    explanation = explainer.explain(sample_answers)
    print(f"Predicted score: {explanation['predicted_score']:.2f}")
    print(f"\nDimension contributions:")
    for dc in explanation["dimension_contributions"]:
        sign = "+" if dc["shap_total"] >= 0 else ""
        print(f"  {dc['dimension_name']:20s}  avg={dc['avg_score']:.2f}  "
              f"SHAP={sign}{dc['shap_total']:.3f}  ({dc['contribution_pct']:.1f}%)")

    print(f"\nTop strengths:")
    for s in explanation["top_strengths"]:
        print(f"  + {s['feature_label']}: value={s['value']:.1f}, shap={s['shap_value']:+.3f}")

    print(f"\nTop weaknesses:")
    for w in explanation["top_weaknesses"]:
        print(f"  - {w['feature_label']}: value={w['value']:.1f}, shap={w['shap_value']:+.3f}")

    print(f"\nWhat-if (improve each dim by +1):")
    for wi in explainer.all_what_ifs(sample_answers, delta=1.0):
        print(f"  improve {wi['dimension_name']:20s} +1 -> "
              f"{wi['current_score']:.2f} -> {wi['new_score']:.2f}  (Δ={wi['delta']:+.2f})")
