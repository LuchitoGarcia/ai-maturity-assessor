FROM python:3.11-slim

WORKDIR /app

# System deps for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY backend/ ./backend/
COPY data/ ./data/

# Generate dataset and train model at build time so the image is self-contained.
# This makes the container fully runnable on first start with no setup.
RUN python backend/ml/synthetic_generator.py && \
    python backend/ml/train.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
