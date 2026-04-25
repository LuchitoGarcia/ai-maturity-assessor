"""
Model Training Module
=====================
Trains an XGBoost regression model to predict AI maturity score
from the 25 questionnaire answers + engineered features.

Pipeline:
1. Load synthetic dataset
2. Feature engineering (raw + derived features)
3. Train/test split (80/20, stratified by sector)
4. 10-fold cross-validation with metrics
5. Final model trained on full data
6. Save model + metrics report
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.framework import FRAMEWORK


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

ALL_QUESTION_IDS = [s.question.id for d in FRAMEWORK for s in d.sub_dimensions]
DIMENSION_IDS = [d.id for d in FRAMEWORK]


def compute_dimension_averages(df: pd.DataFrame) -> pd.DataFrame:
    """For each dimension, compute the average of its question scores."""
    out = df.copy()
    for dim in FRAMEWORK:
        ids = [s.question.id for s in dim.sub_dimensions]
        out[f"avg_{dim.id}"] = df[ids].mean(axis=1)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate derived features.

    These are interpretable features that help the model AND make
    the final analysis richer (consultancy-level insights).
    """
    out = compute_dimension_averages(df)

    # 1) Cross-dimension gaps (key consultancy diagnostics)
    out["gap_strategy_vs_tech"] = out["avg_strategy"] - out["avg_technology"]
    out["gap_data_vs_talent"] = out["avg_data"] - out["avg_talent"]
    out["gap_strategy_vs_processes"] = out["avg_strategy"] - out["avg_processes"]

    # 2) Variance across dimensions: low variance = balanced; high = lopsided
    dim_cols = [f"avg_{d}" for d in DIMENSION_IDS]
    out["dim_variance"] = out[dim_cols].var(axis=1)
    out["dim_min"] = out[dim_cols].min(axis=1)  # weakest dimension
    out["dim_max"] = out[dim_cols].max(axis=1)  # strongest dimension
    out["dim_range"] = out["dim_max"] - out["dim_min"]

    # 3) Internal consistency per dimension (low variance within = consistent)
    for dim in FRAMEWORK:
        ids = [s.question.id for s in dim.sub_dimensions]
        out[f"consistency_{dim.id}"] = -df[ids].var(axis=1)  # negative variance

    # 4) Composite "foundation" score: data + tech + talent
    out["foundation_score"] = (out["avg_data"] + out["avg_technology"] + out["avg_talent"]) / 3

    # 5) Composite "execution" score: strategy + processes
    out["execution_score"] = (out["avg_strategy"] + out["avg_processes"]) / 2

    # 6) Quick-win potential: high strategy but low processes (room to scale)
    out["quick_win_signal"] = np.maximum(0, out["avg_strategy"] - out["avg_processes"])

    return out


def get_feature_columns() -> List[str]:
    """Return the ordered list of features used by the model."""
    cols = list(ALL_QUESTION_IDS)  # 25 raw answers
    cols += [f"avg_{d}" for d in DIMENSION_IDS]  # 5 averages
    cols += [
        "gap_strategy_vs_tech",
        "gap_data_vs_talent",
        "gap_strategy_vs_processes",
        "dim_variance", "dim_min", "dim_max", "dim_range",
        "foundation_score", "execution_score", "quick_win_signal",
    ]
    cols += [f"consistency_{d}" for d in DIMENSION_IDS]  # 5 consistency
    return cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cross_validate(X: pd.DataFrame, y: pd.Series, params: Dict, k: int = 10
                   ) -> Dict:
    """K-fold cross-validation."""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses, maes, r2s = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)

        rmses.append(float(np.sqrt(mean_squared_error(y_va, preds))))
        maes.append(float(mean_absolute_error(y_va, preds)))
        r2s.append(float(r2_score(y_va, preds)))

    return {
        "k": k,
        "rmse": {"mean": float(np.mean(rmses)), "std": float(np.std(rmses)),
                 "per_fold": rmses},
        "mae": {"mean": float(np.mean(maes)), "std": float(np.std(maes)),
                "per_fold": maes},
        "r2": {"mean": float(np.mean(r2s)), "std": float(np.std(r2s)),
               "per_fold": r2s},
    }


def ablation_study(X: pd.DataFrame, y: pd.Series, params: Dict) -> Dict:
    """Train models removing each dimension's questions to measure impact."""
    base = cross_validate(X, y, params, k=5)
    base_r2 = base["r2"]["mean"]

    results = {"baseline_r2": base_r2, "ablations": {}}
    for dim in FRAMEWORK:
        cols_to_drop = [s.question.id for s in dim.sub_dimensions]
        cols_to_drop += [f"avg_{dim.id}", f"consistency_{dim.id}"]
        X_ablated = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
        ablated = cross_validate(X_ablated, y, params, k=5)
        delta = base_r2 - ablated["r2"]["mean"]
        results["ablations"][dim.id] = {
            "r2": ablated["r2"]["mean"],
            "delta_r2": delta,
        }
    return results


def train_final_model(X: pd.DataFrame, y: pd.Series, params: Dict
                      ) -> xgb.XGBRegressor:
    """Train the final model on the full dataset."""
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)
    return model


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "synthetic" / "synthetic_companies.csv"
    out_dir = project_root / "backend" / "ml" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    print(f"  shape: {df.shape}")

    # Feature engineering
    df_feat = engineer_features(df)
    feature_cols = get_feature_columns()
    X = df_feat[feature_cols]
    y = df_feat["maturity_score"]

    # Train/test split (stratified by sector)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=df_feat["sector"]
    )
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

    # XGBoost params (reasonable defaults for tabular regression)
    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    # Cross-validation on train set
    print("\nRunning 10-fold cross-validation...")
    cv_results = cross_validate(X_train, y_train, params, k=10)
    print(f"  RMSE: {cv_results['rmse']['mean']:.4f} ± {cv_results['rmse']['std']:.4f}")
    print(f"  MAE:  {cv_results['mae']['mean']:.4f} ± {cv_results['mae']['std']:.4f}")
    print(f"  R²:   {cv_results['r2']['mean']:.4f} ± {cv_results['r2']['std']:.4f}")

    # Ablation study
    print("\nRunning ablation study (which dimension matters most)...")
    abl = ablation_study(X_train, y_train, params)
    print(f"  Baseline R²: {abl['baseline_r2']:.4f}")
    for dim_id, res in abl["ablations"].items():
        print(f"  - removing {dim_id:13s}: R²={res['r2']:.4f}  Δ={res['delta_r2']:+.4f}")

    # Train final model on train set, evaluate on held-out test
    print("\nTraining final model...")
    model = train_final_model(X_train, y_train, params)
    test_preds = model.predict(X_test)
    test_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
        "mae": float(mean_absolute_error(y_test, test_preds)),
        "r2": float(r2_score(y_test, test_preds)),
    }
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE:  {test_metrics['mae']:.4f}")
    print(f"  Test R²:   {test_metrics['r2']:.4f}")

    # Re-train on FULL dataset for production model
    print("\nRetraining on full dataset for production...")
    final_model = train_final_model(X, y, params)

    # Save artifacts
    joblib.dump(final_model, out_dir / "xgboost_model.pkl")
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(out_dir / "training_params.json", "w") as f:
        json.dump(params, f, indent=2)
    with open(out_dir / "metrics_report.json", "w") as f:
        json.dump({
            "cv_results": cv_results,
            "ablation_study": abl,
            "test_metrics": test_metrics,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(X.shape[1]),
        }, f, indent=2)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    print(f"\nArtifacts saved to: {out_dir}")
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
