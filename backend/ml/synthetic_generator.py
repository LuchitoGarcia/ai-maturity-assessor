"""
Synthetic Dataset Generator
============================
Generates a realistic dataset of companies with coherent AI maturity profiles.

Methodology:
- Each company has a base profile determined by its sector
- Sector-specific priors come from industry research (Deloitte, Gartner)
- Within-company correlations reflect real-world dependencies:
  e.g., high data quality correlates with strong governance
- Controlled noise simulates within-sector variability
- ~5% of companies are designed as "outliers" (atypical profiles)

The synthetic ground-truth `maturity_score` is computed via a weighted
formula based on the framework. The ML model later learns to predict
this from the 25 raw answers (i.e., we test that the model recovers
a known function under noise — a strong validation strategy).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.framework import FRAMEWORK, get_question_to_dimension_map


# ---------------------------------------------------------------------------
# Sector profiles
# Each sector has typical strengths/weaknesses observed in industry studies.
# Format: dimension_id -> mean (1-5 scale)
# ---------------------------------------------------------------------------
SECTOR_PROFILES: Dict[str, Dict[str, float]] = {
    "fintech": {
        # Fintechs are data-mature and tech-modern, but processes can be young
        "data": 3.8,
        "talent": 3.6,
        "technology": 3.9,
        "strategy": 3.5,
        "processes": 3.0,
    },
    "banking_insurance": {
        # Traditional finance: regulated, structured, but tech-debt heavy
        "data": 3.4,
        "talent": 2.8,
        "technology": 2.5,
        "strategy": 3.0,
        "processes": 3.5,
    },
    "manufacturing": {
        # Strong on processes, weak on data culture and AI adoption
        "data": 2.3,
        "talent": 2.2,
        "technology": 2.6,
        "strategy": 2.5,
        "processes": 3.5,
    },
    "retail_ecommerce": {
        # Data-rich (transactions), variable on infrastructure
        "data": 3.5,
        "talent": 3.0,
        "technology": 3.2,
        "strategy": 3.3,
        "processes": 2.8,
    },
    "healthcare": {
        # Strict governance, conservative tech adoption
        "data": 2.8,
        "talent": 2.5,
        "technology": 2.4,
        "strategy": 2.7,
        "processes": 3.2,
    },
    "logistics": {
        # Process-heavy, automation-oriented but data fragmented
        "data": 2.6,
        "talent": 2.4,
        "technology": 2.8,
        "strategy": 2.7,
        "processes": 3.4,
    },
    "energy_utilities": {
        # Capital-intensive, slower adoption but increasing
        "data": 2.7,
        "talent": 2.3,
        "technology": 2.5,
        "strategy": 2.6,
        "processes": 3.0,
    },
    "tech_software": {
        # Highest baseline across the board
        "data": 4.0,
        "talent": 4.2,
        "technology": 4.3,
        "strategy": 3.8,
        "processes": 3.5,
    },
    "professional_services": {
        # Consulting/legal/etc. - heterogeneous
        "data": 2.9,
        "talent": 3.2,
        "technology": 2.8,
        "strategy": 3.0,
        "processes": 3.0,
    },
    "education": {
        # Generally lower maturity, growing
        "data": 2.4,
        "talent": 2.6,
        "technology": 2.3,
        "strategy": 2.5,
        "processes": 2.7,
    },
}

# Company size adjustments (larger = generally more mature in tech & strategy,
# but also more legacy debt)
SIZE_PROFILES: Dict[str, Dict[str, float]] = {
    "startup": {        # <50 employees
        "technology": +0.3, "strategy": -0.2, "processes": -0.4,
        "talent": +0.1, "data": -0.1,
    },
    "small": {          # 50-200
        "technology": +0.1, "strategy": +0.0, "processes": +0.0,
        "talent": +0.0, "data": +0.0,
    },
    "medium": {         # 200-1000
        "technology": +0.0, "strategy": +0.2, "processes": +0.2,
        "talent": +0.1, "data": +0.1,
    },
    "large": {          # 1000-5000
        "technology": -0.1, "strategy": +0.4, "processes": +0.3,
        "talent": +0.2, "data": +0.2,
    },
    "enterprise": {     # >5000
        "technology": -0.2, "strategy": +0.5, "processes": +0.4,
        "talent": +0.3, "data": +0.3,
    },
}

# Within-dimension correlation: questions in the same dimension move together
# (e.g., good data quality → likely good governance)
WITHIN_DIM_CORR = 0.55

# Cross-dimension correlations (subtle but realistic)
CROSS_DIM_CORR = 0.25


def _generate_company(rng: np.random.Generator, sector: str, size: str,
                      is_outlier: bool = False) -> Dict:
    """Generate one synthetic company."""
    sector_profile = SECTOR_PROFILES[sector]
    size_adj = SIZE_PROFILES[size]

    # Per-dimension means after sector + size adjustments
    dim_means = {}
    for dim_id, base in sector_profile.items():
        adjusted = base + size_adj.get(dim_id, 0.0)
        dim_means[dim_id] = np.clip(adjusted, 1.0, 5.0)

    # If outlier: shift a couple of dimensions strongly
    if is_outlier:
        dims_to_shift = rng.choice(list(dim_means.keys()), size=2, replace=False)
        for d in dims_to_shift:
            shift = rng.choice([-1.5, +1.5])
            dim_means[d] = np.clip(dim_means[d] + shift, 1.0, 5.0)

    # Cross-dimension shared factor (a "company tier" effect)
    company_tier = rng.normal(0, 0.4)

    answers = {}
    qmap = get_question_to_dimension_map()

    # Generate per-dimension shared factor (within-dim correlation)
    dim_shared_factors = {
        dim.id: rng.normal(0, 1.0) for dim in FRAMEWORK
    }

    for dim in FRAMEWORK:
        dim_mean = dim_means[dim.id]
        dim_shared = dim_shared_factors[dim.id]

        for sub in dim.sub_dimensions:
            # Compose the latent score:
            # - dim_mean: sector/size baseline
            # - within-dim shared: WITHIN_DIM_CORR factor
            # - cross-dim shared: CROSS_DIM_CORR * tier
            # - per-question idiosyncratic noise
            latent = (
                dim_mean
                + WITHIN_DIM_CORR * dim_shared
                + CROSS_DIM_CORR * company_tier
                + rng.normal(0, 0.7)
            )
            # Quantize to Likert scale 1..5
            score = int(np.clip(np.round(latent), 1, 5))
            answers[sub.question.id] = score

    # Compute ground-truth maturity score: weighted mean of per-dim means
    dim_avgs = {}
    for dim in FRAMEWORK:
        ids = [s.question.id for s in dim.sub_dimensions]
        dim_avgs[dim.id] = np.mean([answers[i] for i in ids])

    overall = sum(dim_avgs[d.id] * d.weight for d in FRAMEWORK)

    return {
        "sector": sector,
        "size": size,
        "is_outlier": is_outlier,
        **answers,
        **{f"dim_avg_{k}": v for k, v in dim_avgs.items()},
        "maturity_score": float(overall),
    }


def generate_dataset(n: int = 1000, seed: int = 42,
                     outlier_pct: float = 0.05) -> pd.DataFrame:
    """Generate the full synthetic dataset."""
    rng = np.random.default_rng(seed)

    sectors = list(SECTOR_PROFILES.keys())
    sizes = list(SIZE_PROFILES.keys())

    # Realistic sector distribution (some sectors are larger)
    sector_weights = np.array([
        0.07,  # fintech
        0.10,  # banking_insurance
        0.15,  # manufacturing
        0.12,  # retail_ecommerce
        0.10,  # healthcare
        0.08,  # logistics
        0.07,  # energy_utilities
        0.13,  # tech_software
        0.12,  # professional_services
        0.06,  # education
    ])
    sector_weights = sector_weights / sector_weights.sum()

    # Size distribution: small/medium most common
    size_weights = np.array([0.20, 0.30, 0.25, 0.15, 0.10])

    rows = []
    n_outliers = int(n * outlier_pct)
    outlier_idx = set(rng.choice(n, size=n_outliers, replace=False).tolist())

    for i in range(n):
        sector = rng.choice(sectors, p=sector_weights)
        size = rng.choice(sizes, p=size_weights)
        is_outlier = i in outlier_idx
        row = _generate_company(rng, sector, size, is_outlier)
        rows.append(row)

    return pd.DataFrame(rows)


def save_dataset_with_metadata(df: pd.DataFrame, out_dir: Path):
    """Save dataset + metadata for reproducibility."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the dataset
    df.to_csv(out_dir / "synthetic_companies.csv", index=False)

    # Sector benchmarks (means + std per sector)
    benchmarks = {}
    for sector in df["sector"].unique():
        sub = df[df["sector"] == sector]
        benchmarks[sector] = {
            "n": int(len(sub)),
            "mean_score": float(sub["maturity_score"].mean()),
            "std_score": float(sub["maturity_score"].std()),
            "p25": float(sub["maturity_score"].quantile(0.25)),
            "p50": float(sub["maturity_score"].quantile(0.50)),
            "p75": float(sub["maturity_score"].quantile(0.75)),
            "dim_means": {
                d.id: float(sub[f"dim_avg_{d.id}"].mean())
                for d in FRAMEWORK
            },
        }
    with open(out_dir / "sector_benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=2)

    # Metadata
    meta = {
        "n_companies": int(len(df)),
        "n_sectors": int(df["sector"].nunique()),
        "n_sizes": int(df["size"].nunique()),
        "n_outliers": int(df["is_outlier"].sum()),
        "score_stats": {
            "mean": float(df["maturity_score"].mean()),
            "std": float(df["maturity_score"].std()),
            "min": float(df["maturity_score"].min()),
            "max": float(df["maturity_score"].max()),
        },
        "framework_dimensions": [d.id for d in FRAMEWORK],
        "n_questions": sum(len(d.sub_dimensions) for d in FRAMEWORK),
        "generation_params": {
            "within_dim_corr": WITHIN_DIM_CORR,
            "cross_dim_corr": CROSS_DIM_CORR,
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_dataset(n=1000, seed=42)

    out = Path(__file__).resolve().parent.parent.parent / "data" / "synthetic"
    save_dataset_with_metadata(df, out)

    print(f"\nDataset saved to: {out}")
    print(f"Shape: {df.shape}")
    print(f"\nScore distribution by sector:")
    print(df.groupby("sector")["maturity_score"].agg(["mean", "std", "count"]).round(2))
    print(f"\nOverall: mean={df['maturity_score'].mean():.2f}, "
          f"std={df['maturity_score'].std():.2f}")
