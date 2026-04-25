# AI Maturity Assessor

> A research-grounded assessment tool that scores organizational AI readiness across 5 dimensions, explains *why* using SHAP, and generates a personalized phased roadmap.

[![Python](https://img.shields.io/badge/python-3.11+-blue)]() [![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]() [![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## Overview

The AI Maturity Assessor is an end-to-end system that quantifies how prepared an organization is to adopt and scale AI. It goes beyond a single number: it produces an **explainable score**, **sector-relative benchmarks**, **what-if simulations**, and a **prioritized 18-month roadmap** of concrete initiatives.

The project combines a calibrated ML model (XGBoost), modern explainability (SHAP), and a deterministic rule-based roadmap engine — all served by a FastAPI backend and consumed by a single-page React frontend.

### Key features

| | |
|---|---|
| **Calibrated ML model** | XGBoost regression with 10-fold cross-validated R² > 0.99, trained on a 1,000-company synthetic benchmark |
| **Explainability** | Per-feature and per-dimension SHAP attributions, top strengths/weaknesses, what-if simulations |
| **Sector benchmarks** | Percentile ranking against peers in 10 industry sectors |
| **Personalized roadmap** | 3-phase plan (0–3 mo, 3–9 mo, 9–18 mo) drawn from a knowledge base of 25+ initiatives |
| **No black box** | All scoring, weighting, and roadmap rules are documented and inspectable |

---

## Methodology

The framework synthesizes four established sources:

- **Gartner AI Maturity Model** — five-level maturity scale (Initial → Optimizing).
- **MIT Sloan Digital Maturity Framework** (Westerman et al.) — multi-dimensional capability assessment.
- **Deloitte State of AI in the Enterprise** (annual reports) — empirical priors on sector adoption patterns.
- **Harvard Business Review** — *Building the AI-Powered Organization* (Fountaine, McCarthy & Saleh, 2019) — the importance of executive sponsorship and cultural readiness.

### The 5 dimensions

| # | Dimension | Weight | Sub-dimensions |
|---|-----------|--------|----------------|
| 1 | **Data** | 25% | Quality · Governance · Accessibility · Integration · Volume |
| 2 | **Talent & Culture** | 20% | Skills · Data-driven culture · Upskilling · Collaboration · Change readiness |
| 3 | **Technology** | 20% | Cloud · MLOps · Analytics · Tech debt · Security |
| 4 | **Strategy** | 20% | C-suite vision · Budget · Use cases · KPIs · Ethics |
| 5 | **Processes** | 15% | Automation · Documentation · Continuous improvement · AI integration · Feedback loops |

Each sub-dimension is measured by one Likert-scale question (1–5). Total: **25 questions**.

### Maturity levels

| Level | Score range | Description |
|-------|-------------|-------------|
| **Initial** | 0.0 – 1.5 | AI is largely absent; decisions rely on intuition |
| **Exploring** | 1.5 – 2.5 | Small pilots, growing awareness |
| **Developing** | 2.5 – 3.5 | Multiple initiatives, foundations forming |
| **Scaling** | 3.5 – 4.3 | AI produces measurable business value at scale |
| **Optimizing** | 4.3 – 5.0 | AI is core to competitive advantage |

---

## Architecture

```
┌──────────────────┐    REST API    ┌──────────────────────┐
│  React Frontend  │ ──────────────▶│   FastAPI Backend    │
│  (single HTML)   │                 │                      │
└──────────────────┘                 └─────┬────────────────┘
                                           │
                ┌──────────────────────────┼──────────────────────────┐
                ▼                          ▼                          ▼
       ┌─────────────────┐       ┌──────────────────┐      ┌──────────────────┐
       │  XGBoost Model  │       │   SHAP Engine    │      │ Roadmap Engine   │
       │   (45 features) │       │  (TreeExplainer) │      │  (rule-based)    │
       └─────────────────┘       └──────────────────┘      └──────────────────┘
                ▲                          ▲                          ▲
                │                          │                          │
                └──────── Synthetic Dataset (1000 companies) ─────────┘
```

### Modeling pipeline

1. **Synthetic dataset generation** (`backend/ml/synthetic_generator.py`)
   - 1,000 companies with sector-aware profiles (10 sectors, 5 size buckets).
   - Realistic within-dimension correlations (0.55) and cross-dimension shared factor (0.25).
   - 5% outliers injected to test robustness.
2. **Feature engineering** (`backend/ml/train.py`)
   - 25 raw answers → 45 features (raw + dimension averages + cross-dim gaps + variance + composites + consistency scores).
3. **Training**
   - 80/20 train/test split stratified by sector.
   - XGBoost with 10-fold cross-validation.
   - Ablation study: removes each dimension to measure its predictive contribution.
4. **Explainability** (`backend/ml/explainer.py`)
   - `shap.TreeExplainer` for fast per-prediction attributions.
   - Per-dimension aggregation (sum of related-feature SHAP values).
   - What-if simulation: re-predicts with one dimension shifted ±n.

---

## Quick start

### 1. Install

```bash
git clone <repo-url>
cd ai-maturity-assessor
pip install -r requirements.txt
```

### 2. Generate the dataset and train the model

```bash
python backend/ml/synthetic_generator.py     # generates data/synthetic/
python backend/ml/train.py                    # trains and saves backend/ml/artifacts/
```

### 3. Run the backend

```bash
python -m uvicorn backend.api.app:app --reload --port 8000
```

The API is now live at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### 4. Open the frontend

Open `frontend/index.html` in any modern browser. It will auto-detect localhost and connect to the backend.

For a production deployment, serve the file via any static server.

### 5. Run tests

```bash
python -m pytest backend/tests/ -v
```

All 31 tests should pass.

### 6. Explore the notebooks

```bash
cd notebooks/
jupyter notebook
```

Three notebooks walk through:
- `01_exploratory_data_analysis.ipynb` — dataset inspection.
- `02_model_training.ipynb` — CV results, ablation, feature importance.
- `03_shap_analysis.ipynb` — global and local SHAP explanations.

### Docker (optional)

```bash
docker compose up
```

This starts the backend on port 8000. Open `frontend/index.html` separately.

---

## API reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Health check (model + benchmarks loaded) |
| GET | `/api/framework` | Full framework definition (dimensions + questions) |
| POST | `/api/assess` | Score an assessment + benchmark |
| POST | `/api/explain` | SHAP explanation (top features + dimension contributions) |
| POST | `/api/whatif` | Simulate improving one dimension |
| POST | `/api/whatif/all` | What-if for all dimensions |
| POST | `/api/roadmap` | Generate phased roadmap |
| GET | `/api/benchmark/{sector}` | Sector benchmark statistics |
| POST | `/api/full_report` | Single call: assess + explain + what-if + roadmap |

### Example request

```bash
curl -X POST http://localhost:8000/api/full_report \
  -H "Content-Type: application/json" \
  -d '{
    "answers": {
      "d1_q1": 4, "d1_q2": 3, "d1_q3": 4, "d1_q4": 4, "d1_q5": 3,
      "d2_q1": 2, "d2_q2": 3, "d2_q3": 2, "d2_q4": 3, "d2_q5": 3,
      "d3_q1": 4, "d3_q2": 3, "d3_q3": 4, "d3_q4": 4, "d3_q5": 4,
      "d4_q1": 4, "d4_q2": 3, "d4_q3": 3, "d4_q4": 3, "d4_q5": 3,
      "d5_q1": 3, "d5_q2": 3, "d5_q3": 3, "d5_q4": 2, "d5_q5": 2
    },
    "sector": "manufacturing",
    "company_size": "medium"
  }'
```

---

## Model performance

Trained on 800 companies, evaluated on a held-out 200-company test set:

| Metric | Cross-validation (10-fold) | Held-out test |
|--------|----------------------------|---------------|
| RMSE | 0.0400 ± 0.0036 | 0.0365 |
| MAE | 0.0285 ± 0.0022 | 0.0276 |
| R² | 0.9949 ± 0.0009 | 0.9955 |

**Note on R².** The synthetic ground-truth score is a deterministic function of the dimension averages plus controlled noise. The high R² indicates the model successfully recovers the underlying scoring function. The ablation study confirms each dimension carries independent signal — removing any single dimension degrades R² in a predictable, measurable way.

### Top features (XGBoost gain)

```
foundation_score          49.1%   (composite: data + tech + talent)
dim_min                   23.5%   (weakest dimension)
dim_max                   12.1%   (strongest dimension)
execution_score            6.5%   (composite: strategy + processes)
avg_data                   2.9%
avg_strategy               2.3%
... others <1% each
```

---

## Project structure

```
ai-maturity-assessor/
├── backend/
│   ├── api/                  # FastAPI app
│   ├── ml/                   # Framework, dataset gen, training, SHAP
│   │   └── artifacts/        # Trained model + metrics
│   ├── llm/                  # Roadmap engine (deterministic + LLM-ready)
│   └── tests/                # 31 pytest tests
├── frontend/
│   └── index.html            # Single-file React app (CDN, no build)
├── data/
│   └── synthetic/            # Generated dataset + sector benchmarks
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_shap_analysis.ipynb
├── docs/                     # Methodology, white paper
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Limitations & future work

The current MVP is a calibrated, explainable, and reproducible system, but it has known limitations that the next iterations will address:

- **Synthetic data**: real benchmark data per sector is the natural next step. The framework and pipeline are ready to ingest it.
- **Cross-cultural validity**: the question wording and scale labels are calibrated to English-language Western contexts.
- **LLM-augmented narratives**: the roadmap engine is deterministic; an optional LLM layer can enrich initiative descriptions with sector-specific examples (the hook is already in place — see `backend/llm/roadmap.py::augment_with_llm`).
- **Empirical validation**: a planned next step is a longitudinal study with at least one real organization to validate the predictive value of the score over time.

---

## References

- Davenport, T. H. & Ronanki, R. (2018). *Artificial Intelligence for the Real World*. Harvard Business Review.
- Deloitte (2023). *State of AI in the Enterprise* — annual survey reports.
- Fountaine, T., McCarthy, B. & Saleh, T. (2019). *Building the AI-Powered Organization*. Harvard Business Review.
- Gartner (2022). *Gartner AI Maturity Model*. Gartner Research.
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Westerman, G., Bonnet, D. & McAfee, A. (2014). *Leading Digital: Turning Technology into Business Transformation*. Harvard Business Press.

---

## License

MIT — see `LICENSE`.
