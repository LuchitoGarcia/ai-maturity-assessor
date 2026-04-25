.PHONY: install data train test serve all docker-build docker-up clean

install:
	pip install -r requirements.txt

data:
	python backend/ml/synthetic_generator.py

train:
	python backend/ml/train.py

test:
	python -m pytest backend/tests/ -v

serve:
	python -m uvicorn backend.api.app:app --reload --port 8000

all: install data train test
	@echo ""
	@echo "✅ Setup complete. Run 'make serve' to start the backend,"
	@echo "   then open frontend/index.html in your browser."

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf backend/ml/artifacts/*.pkl
	rm -rf backend/ml/artifacts/*.json
	rm -rf backend/ml/artifacts/*.csv
	rm -rf data/synthetic/*
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name ".pytest_cache" -exec rm -rf {} +
