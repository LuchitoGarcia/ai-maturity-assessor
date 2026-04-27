# AI Maturity Assessor

> A research-grounded assessment tool that scores organizational AI readiness across 5 dimensions, explains *why* using SHAP, and generates a personalized phased roadmap.

[![Python](https://img.shields.io/badge/python-3.11+-blue)]() [![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]() [![License](https://img.shields.io/badge/license-MIT-blue)]()

**Live demo**: https://ai-maturity-assessor-production.up.railway.app

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
| **Personalized roadmap** | 3-phase plan (0–3 mo, 3–9 mo, 9–18 mo) with timeline and estimated ROI |
| **AI Advisor** | Chatbot that answers strategic questions about your results |
| **No signup required** | 25 questions · ~6 minutes · instant results |

---

## Quick start (Local Development)

### 1. Clone and navigate

```bash
cd /Users/luisgarciaalvarez/Documents/ai-maturity-assessor
```

### 2. Activate the virtual environment

```bash
conda deactivate
source venv_ma/bin/activate
```

### 3. Start the backend (in one terminal)

```bash
python -m uvicorn backend.api.app:app --port 8000 &
```

The API will be live at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### 4. Start the frontend (in another terminal)

```bash
cd frontend
python -m http.server 3000
```

### 5. Open in your browser

```
http://localhost:3000
```

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

## Model performance

| Metric | Cross-validation | Test set |
|--------|------------------|----------|
| RMSE | 0.0400 ± 0.0036 | 0.0365 |
| MAE | 0.0285 ± 0.0022 | 0.0276 |
| R² | 0.9949 ± 0.0009 | 0.9955 |

---

## Alternative: Docker

If you prefer containerization:

```bash
docker compose up
```

This starts the backend on port 8000. Open `frontend/index.html` in your browser or serve via the http.server command above.

---

## Stack

- **Backend**: Python 3.11 · FastAPI · XGBoost · SHAP · Scikit-learn
- **Frontend**: React 18 · Tailwind CSS · Chart.js (via CDN)
- **ML**: Pandas · NumPy · Joblib
- **Deploy**: Railway · Docker

---

## Project structure

```
ai-maturity-assessor/
├── backend/
│   ├── api/                  # FastAPI endpoints
│   ├── ml/                   # ML pipeline, SHAP explainer
│   ├── llm/                  # AI advisor engine
│   └── tests/                # 31 pytest tests
├── frontend/
│   └── index.html            # Single-file React app
├── data/
│   └── synthetic/            # Generated dataset
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_shap_analysis.ipynb
├── docs/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## References

- Gartner (2022). *Gartner AI Maturity Model*.
- Westerman, G., Bonnet, D. & McAfee, A. (2014). *Leading Digital*.
- Fountaine, T., McCarthy, B. & Saleh, T. (2019). *Building the AI-Powered Organization*. HBR.
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.

---

## License

MIT — see `LICENSE`.

---

## Author

Built by [Luisito García Álvarez](https://linkedin.com/in/luisito-garcia) · AI Student · Accenture