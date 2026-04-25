# Methodology

This document explains the design choices behind the AI Maturity Assessor framework, with citations and rationale.

---

## Why a five-dimension framework?

Most academic and industry frameworks for organizational AI/digital maturity converge on between three and seven dimensions. Five is a sweet spot:

- **Three** is too coarse to be diagnostic.
- **Seven or more** introduces redundancy and respondent fatigue.
- **Five** maps cleanly onto established practice (Gartner, MIT Sloan, Deloitte all use 5–6 dimensions).

The five dimensions are:

1. **Data** — the substrate on which everything else rests.
2. **Talent & Culture** — who can build it and whether they are listened to.
3. **Technology** — the platforms and pipelines that scale ML.
4. **Strategy** — the C-suite alignment and resourcing.
5. **Processes** — the operational backbone that turns models into outcomes.

---

## Why these specific weights?

The weights (0.25, 0.20, 0.20, 0.20, 0.15) reflect a synthesis of the literature:

- **Data is weighted highest (0.25)** because every other dimension is constrained by it. A consensus across Gartner and Deloitte reports identifies poor data as the most-cited barrier to AI value capture.
- **Three dimensions (Talent, Technology, Strategy) at 0.20** because each is roughly co-equal in the practitioner literature: Fountaine et al. (2019) emphasize talent and strategy as critical; Davenport (2018) emphasizes technology and strategy.
- **Processes weighted lowest (0.15)** because process maturity is somewhat *downstream* of the other four — better data, talent, technology, and strategy all push process maturity up.

These weights are an informed prior. A planned follow-up will calibrate them empirically using regression of multi-respondent organizational data against business outcomes (see WHITE_PAPER §10).

---

## Why anchored Likert scales?

Each question uses a 5-point scale with a *labeled anchor at every point*, not just at the endpoints. The survey methodology literature is consistent on this:

- Endpoint-only scales force respondents to interpolate, introducing inter-respondent variance.
- Fully anchored scales reduce that variance and make answers more comparable.
- 5 points balances expressiveness with cognitive load (7+ points are noisy; 3 points are too coarse).

Example anchor for "data quality":
- 1: Inconsistent and difficult to use
- 2: Usable but requires frequent manual cleaning
- 3: Reasonably reliable for basic analysis
- 4: High quality with established validation processes
- 5: Excellent, governed, with automated pipelines

A respondent reading these can make a more reproducible judgment than one asked simply "rate your data quality from 1 to 5."

---

## Why XGBoost?

For this problem (tabular, 25–45 features, 1,000 rows, regression), XGBoost is well-suited because:

- Handles small datasets with regularization.
- Captures non-linear interactions between dimensions (e.g., the interaction between strategy and technology — high strategy with low tech yields a different score than the additive prediction).
- Provides fast SHAP TreeExplainer for downstream explainability.
- Industry-standard, reproducible, and interpretable.

Alternative architectures (linear regression, random forest, neural net) were considered:

- **Linear regression** would underfit the interactions we care about (e.g., the "lopsided profile" effect).
- **Random forest** is comparable but provides slightly less expressive interaction modeling.
- **Neural networks** are overkill for 1,000 samples and 45 features, and their explainability story is weaker.

---

## Why synthetic data?

Real cross-organizational AI maturity data at scale is not publicly available. Building a real benchmark dataset is a multi-year effort that requires partnerships, NDAs, and standardized data collection.

The synthetic dataset is a deliberate methodological choice that lets us:

1. **Validate the pipeline end-to-end** without waiting for real data.
2. **Encode known industry priors** (sector means, size effects) so the model behaves sensibly on plausible inputs.
3. **Test robustness** by injecting known outliers and known correlation structure.
4. **Provide reproducibility** — anyone can regenerate the dataset deterministically with the seed.

The known limitation is that the model's *absolute* predictions reflect the synthetic priors, not real-world ground truth. The *relative* rankings (this profile is more mature than that profile) are likely robust because they are driven by the framework structure, not the synthetic noise.

When real data is collected, the same pipeline can be retrained with no architectural changes. The framework, scoring formula, SHAP layer, and roadmap engine all remain valid.

---

## Why SHAP for explainability?

SHAP (SHapley Additive exPlanations, Lundberg & Lee, 2017) was chosen because:

- **Theoretical grounding**: SHAP values are the unique attribution method satisfying local accuracy, missingness, and consistency.
- **Tree-specific algorithm**: `TreeExplainer` computes exact SHAP values for tree ensembles in polynomial time, making per-prediction explanation cheap.
- **Compatibility with aggregation**: per-feature SHAP values can be summed by group (e.g., per-dimension) and the result remains a valid attribution.
- **Standard in industry**: SHAP is the dominant explainability framework for tree-based models and is widely understood.

Alternatives (LIME, integrated gradients, permutation importance) were considered:

- **LIME** is local-linear and noisier; less stable for production reports.
- **Integrated gradients** is designed for differentiable models, not trees.
- **Permutation importance** is a global measure, not per-prediction.

---

## Why a deterministic roadmap engine?

The roadmap engine maps `(dimension, maturity_level) → list of initiatives` from a curated knowledge base. We chose this over an LLM-generated roadmap for the MVP because:

1. **Reproducibility**: same input → same output. Critical for a portfolio piece.
2. **No API dependency**: works offline, no costs, no rate limits.
3. **Auditability**: every initiative recommendation can be traced to a specific cell in the knowledge base.
4. **Quality control**: the initiatives are written once, reviewed, and reused.

An optional LLM augmentation hook (`augment_with_llm` in `backend/llm/roadmap.py`) is in place for future work to enrich descriptions with sector-specific examples without changing the core deterministic structure.

---

## Maturity-level boundaries

| Level | Range | Rationale |
|-------|------:|-----------|
| Initial | [1.0, 1.5) | A score < 1.5 means the *median* answer was 1 — no AI activity. |
| Exploring | [1.5, 2.5) | Median answer is 2 — some pilots, lots of gaps. |
| Developing | [2.5, 3.5) | Median answer is 3 — multiple initiatives, mixed maturity. |
| Scaling | [3.5, 4.3) | Above-median; scoring 4 on most dimensions. |
| Optimizing | [4.3, 5.0] | Top tier; near-perfect scoring across the board. |

The "Developing" band is the widest because that is where most real organizations sit — the framework should be able to discriminate within that band, which is why the model uses 25 questions and 45 features rather than just averaging dimensions.

---

## Sources

The full reference list is in `WHITE_PAPER.md`. Key sources for the methodology choices above:

- Anchored Likert scales: Krosnick & Presser, *Question and Questionnaire Design*, 2010.
- SHAP theoretical properties: Lundberg & Lee, *NeurIPS 2017*.
- XGBoost: Chen & Guestrin, *KDD 2016*.
- AI maturity dimensions: Gartner, MIT Sloan, Deloitte, Fountaine et al. (HBR 2019).
