# A Framework for Measuring Organizational AI Maturity:
## Methodology, Model, and Explainability

**Version 0.1 — MVP white paper**

---

## Abstract

We present a framework and operational system for measuring the AI maturity of an organization — that is, how prepared it is to adopt and scale artificial intelligence. The framework synthesizes four established sources (Gartner, MIT Sloan, Deloitte, Harvard Business Review) into five weighted dimensions and 25 sub-dimensions, each measured by a Likert-scale question. We train an XGBoost regression model on a synthetic dataset of 1,000 companies generated under sector- and size-specific priors, with controlled within- and cross-dimension correlations. The model achieves R² = 0.995 on a held-out test set. We layer SHAP-based explainability to attribute predictions to specific features and dimensions, and we provide a what-if engine for sensitivity analysis. Finally, we generate a phased 18-month roadmap from a knowledge base of 25 initiatives mapped to dimension-level maturity. The system is exposed via a FastAPI backend and a single-page React frontend, packaged for reproducibility.

---

## 1. Introduction

Investment in AI continues to outpace organizational readiness to absorb it. Industry surveys consistently report a wide gap between the share of companies that experiment with AI and the much smaller share that capture meaningful business value. The gap is rarely about model quality. It is about the surrounding capabilities: data infrastructure, talent, processes, and strategic clarity.

A *maturity assessment* serves three purposes:

1. **Diagnostic** — identify where an organization stands relative to a structured framework.
2. **Comparative** — benchmark against peers in the same sector and size band.
3. **Prescriptive** — translate the diagnostic into concrete next steps.

Most existing assessments stop at step 1, sometimes step 2, and almost never connect the score to actionable recommendations. The goal of this work is to integrate all three steps into one explainable, reproducible system.

---

## 2. Related work

**Gartner AI Maturity Model.** Five levels (Aware, Active, Operational, Systemic, Transformational) covering vision, adoption, value capture, organization, and platforms. Useful as a coarse rubric but not directly operational.

**MIT Sloan Digital Maturity Framework** (Westerman et al.). Multi-dimensional and empirically grounded; emphasizes the importance of *both* digital intensity and transformation management intensity.

**Deloitte State of AI in the Enterprise.** Annual surveys provide empirical priors on adoption patterns, sector differences, and the gap between leaders and laggards.

**Building the AI-Powered Organization** (Fountaine, McCarthy & Saleh, 2019). Identifies executive sponsorship and cross-functional collaboration as the strongest predictors of AI success.

Our framework synthesizes these into a single five-dimension model with explicit weights, a 25-question instrument, and an executable scoring pipeline.

---

## 3. Framework

### 3.1 Dimensions and weights

| # | Dimension | Weight | Rationale for weight |
|---|-----------|-------:|----------------------|
| 1 | **Data** | 0.25 | Data is the substrate; without it, every other capability is constrained. |
| 2 | **Talent & Culture** | 0.20 | Talent and a data-driven culture determine velocity and adoption. |
| 3 | **Technology** | 0.20 | Modern infrastructure (cloud, MLOps) enables scale; legacy debt is a structural drag. |
| 4 | **Strategy** | 0.20 | Without C-suite alignment and budget, initiatives stall. |
| 5 | **Processes** | 0.15 | Process maturity translates models into outcomes; weighted slightly lower because it is downstream of the others. |

Weights sum to 1.0 by construction. They are derived from a qualitative reading of the cited literature; future work will calibrate them empirically.

### 3.2 Sub-dimensions and questions

Each dimension contains five sub-dimensions, each measured by a single Likert-scale question (1–5) with anchored labels at every point of the scale (not just endpoints). This anchoring choice is informed by the survey-design literature: anchored scales produce more comparable inter-respondent ratings than unanchored ones.

The full set of 25 questions is in `backend/ml/framework.py`.

### 3.3 Maturity levels

| Level | Range | Interpretation |
|-------|------:|----------------|
| Initial | [1.0, 1.5) | AI is largely absent; intuition-driven decisions. |
| Exploring | [1.5, 2.5) | Pilots underway; awareness rising. |
| Developing | [2.5, 3.5) | Multiple initiatives; foundations forming. |
| Scaling | [3.5, 4.3) | AI delivers measurable value at scale. |
| Optimizing | [4.3, 5.0] | AI is core to competitive advantage. |

The boundaries are inspired by the Gartner five-level scale, with slightly tighter middle bands to distinguish "developing" from "scaling" companies — the band where most real organizations sit.

---

## 4. Synthetic dataset

We generate 1,000 companies under the following data-generating process:

1. **Sector profile.** For each of 10 sectors, we define base means per dimension (e.g., fintech: high data and tech; manufacturing: high processes, low data).
2. **Size adjustment.** Five size buckets (startup → enterprise) shift the dimension means to reflect typical patterns (e.g., startups higher in technology agility, lower in process maturity).
3. **Outliers.** 5% of companies have two dimensions shifted by ±1.5 to simulate atypical organizations (e.g., a manufacturer with surprisingly mature data).
4. **Correlation structure.**
   - *Within-dimension*: 0.55 shared factor — questions in the same dimension correlate.
   - *Cross-dimension*: 0.25 shared "company tier" factor — overall capability matters.
   - *Idiosyncratic noise*: σ = 0.7 per question.
5. **Quantization.** Latent scores are clipped and rounded to integer values in [1, 5].
6. **Ground truth.** The maturity score is the weighted mean of dimension averages — known and deterministic.

This design lets us validate the model in two senses: it should (a) accurately recover the ground-truth function, and (b) attribute importance correctly to the relevant features.

### 4.1 Resulting distribution by sector

| Sector | Mean | Std | n |
|--------|-----:|----:|--:|
| tech_software | 4.03 | 0.29 | 134 |
| fintech | 3.68 | 0.33 | 75 |
| retail_ecommerce | 3.30 | 0.34 | 116 |
| banking_insurance | 3.10 | 0.33 | 122 |
| professional_services | 3.04 | 0.33 | 102 |
| logistics | 2.83 | 0.31 | 61 |
| energy_utilities | 2.76 | 0.29 | 65 |
| healthcare | 2.75 | 0.32 | 105 |
| manufacturing | 2.64 | 0.33 | 162 |
| education | 2.60 | 0.37 | 58 |

Overall: mean 3.11, std 0.57. The ordering aligns with the qualitative priors of industry surveys.

---

## 5. Model

### 5.1 Feature engineering

The 25 raw answers are extended with derived features designed to be both statistically useful and consultancy-relevant:

- **Dimension averages** (5 features) — `avg_data`, `avg_talent`, etc.
- **Cross-dimension gaps** (3 features) — e.g., `gap_strategy_vs_tech` flags organizations whose ambition outruns their infrastructure.
- **Cross-dim variance / min / max / range** (4 features) — measure how lopsided the profile is.
- **Composite scores** (2 features) — `foundation_score` (data + tech + talent), `execution_score` (strategy + processes).
- **Quick-win signal** (1 feature) — strategy minus processes, capped at zero.
- **Per-dimension consistency** (5 features) — negative variance within each dimension; high consistency means coherent answers.

Total: **45 features** from 25 raw answers.

### 5.2 Architecture

XGBoost regression with the following hyperparameters:

```
n_estimators=400, max_depth=5, learning_rate=0.05,
subsample=0.85, colsample_bytree=0.85,
reg_alpha=0.1, reg_lambda=1.0, tree_method=hist
```

These values were chosen as reasonable defaults for tabular regression of this size; full hyperparameter tuning is a documented next step but not necessary for the current performance level.

### 5.3 Validation

| Metric | 10-fold CV | Held-out test |
|--------|-----------:|--------------:|
| RMSE | 0.0400 ± 0.0036 | 0.0365 |
| MAE | 0.0285 ± 0.0022 | 0.0276 |
| R² | 0.9949 ± 0.0009 | 0.9955 |

Train/test split is stratified by sector to ensure each sector is fairly represented in both partitions.

### 5.4 Ablation study

We retrain the model with each dimension's questions and derived features removed, measuring the drop in R²:

| Dimension removed | R² | Δ R² |
|-------------------|---:|-----:|
| None (baseline) | 0.9949 | — |
| data | 0.9935 | −0.0014 |
| talent | 0.9941 | −0.0008 |
| strategy | 0.9940 | −0.0009 |
| processes | 0.9943 | −0.0006 |
| technology | 0.9947 | −0.0002 |

Every dimension contributes positive predictive signal, confirming the framework is non-redundant.

### 5.5 Note on R²

Because the synthetic ground-truth score is a deterministic function of the dimension averages plus controlled noise, a high R² is the *expected* outcome — it indicates the model has successfully recovered the underlying scoring function from the noisy questionnaire. This is a known property of the synthetic-validation approach. The genuine value of the model is in the engineered features, the SHAP attributions, and the calibrated extrapolation to unseen patterns. When real benchmark data becomes available, R² will drop to a more typical range (likely 0.6–0.8), and additional capacity for noise modeling can be added.

---

## 6. Explainability

We use `shap.TreeExplainer` to attribute each prediction to its constituent features. SHAP values have two desirable properties: (1) they sum to the deviation from the model's expected value (additivity), and (2) they satisfy local accuracy and consistency.

For business consumption, we aggregate per-feature SHAP values into:

- **Per-dimension SHAP totals** — sum of all feature SHAP values whose origin is that dimension.
- **Top strengths** — features with the largest positive SHAP contribution.
- **Top weaknesses** — features with the largest negative SHAP contribution.

### 6.1 What-if simulation

Given a current set of answers, we re-predict with one dimension shifted by ±n points (clipped to [1, 5]). The score delta is the predicted gain from improving that dimension. This is a model-grounded sensitivity analysis: it respects the learned interactions between features rather than assuming linearity.

The "where to invest first" panel in the frontend ranks dimensions by predicted score gain from a +1 improvement.

---

## 7. Roadmap engine

A knowledge base of 25 initiatives is organized as `dimension × maturity_level → list of initiatives`. Each initiative has metadata: title, description, effort (low/medium/high/very high), duration in weeks, ROI category, and intended phase (1, 2, or 3).

Given the per-dimension scores:

1. Sort dimensions by ascending score (weakest first → highest priority).
2. For each dimension, select between 1 and 3 initiatives based on its current level.
3. Group selected initiatives by phase (0–3 mo, 3–9 mo, 9–18 mo).
4. **Promotion rule**: if Phase 1 ends up empty (typical for higher-maturity organizations), promote up to 3 low/medium-effort Phase 2 initiatives from the two weakest dimensions.
5. Generate an executive summary referencing the strongest and weakest dimensions and the sector context.

The engine is fully deterministic. An optional LLM augmentation hook is in place to enrich initiative descriptions with sector-specific examples when an API key is available.

---

## 8. System architecture

The backend is a FastAPI application exposing nine endpoints (see README §API reference). It loads the trained model and SHAP explainer lazily, and reads sector benchmarks from a JSON file.

The frontend is a single HTML file using React, Tailwind, and Recharts via CDN — no build step. It auto-detects localhost vs. production and adjusts the API base URL.

Tests cover the framework (boundaries, weights, mappings — 9 tests), the model (predict range, monotonicity, what-if, structure — 9 tests), and the API (every endpoint plus validation errors — 13 tests). All 31 tests pass.

---

## 9. Limitations

1. **Synthetic data.** The current model is trained on plausible but synthetic priors. Real cross-organizational benchmark data is the obvious next ingredient.
2. **Self-reported measurement.** Like any maturity instrument, this one depends on the candor and calibration of the respondent. Mitigations include anchored scale labels, a "consistency" feature flagged in SHAP, and the recommendation to triangulate across multiple respondents.
3. **Cross-cultural validity.** The instrument is currently English-only with anchors calibrated to Western corporate vocabulary. Translations will require careful re-anchoring rather than literal translation.
4. **Causality.** The model identifies *associations* between scores and recommendations, not causal effects. The roadmap engine encodes practitioner consensus, not validated causal interventions.

---

## 10. Future work

- **Empirical validation** with a small set of real organizations (initially 3–5), including a longitudinal component that tracks score evolution against business outcomes over 6–12 months.
- **Hyperparameter tuning** with Optuna or similar, once we have empirical signal worth fitting to.
- **LLM-enriched roadmaps** with sector-specific case studies and resource recommendations, drawing from a curated knowledge base.
- **Model card and ethics statement** documenting intended use, known biases, and recommended interpretation guidelines.
- **Multi-respondent aggregation** with disagreement signals as a separate diagnostic axis.

---

## References

- Davenport, T. H. & Ronanki, R. (2018). Artificial Intelligence for the Real World. *Harvard Business Review*, January–February 2018.
- Deloitte Insights (2023). State of AI in the Enterprise — Fifth Edition.
- Fountaine, T., McCarthy, B. & Saleh, T. (2019). Building the AI-Powered Organization. *Harvard Business Review*, July–August 2019.
- Gartner (2022). Gartner AI Maturity Model.
- Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
- Westerman, G., Bonnet, D. & McAfee, A. (2014). *Leading Digital: Turning Technology into Business Transformation*. Harvard Business Press.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
