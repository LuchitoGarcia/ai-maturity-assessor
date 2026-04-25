#!/usr/bin/env bash
# Quick setup script: install deps, generate dataset, train model, run tests
set -e

echo "==========================================="
echo "  AI Maturity Assessor — Setup"
echo "==========================================="

echo ""
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/4] Generating synthetic dataset..."
python backend/ml/synthetic_generator.py

echo ""
echo "[3/4] Training XGBoost model..."
python backend/ml/train.py

echo ""
echo "[4/4] Running tests..."
python -m pytest backend/tests/ -v

echo ""
echo "==========================================="
echo "  ✅ Setup complete!"
echo "==========================================="
echo ""
echo "Start the backend:"
echo "  python -m uvicorn backend.api.app:app --reload --port 8000"
echo ""
echo "Then open frontend/index.html in your browser."
