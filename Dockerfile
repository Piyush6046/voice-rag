FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (e.g. git, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary folder structure and copy files needed for model pre-download
RUN mkdir -p backend/pipeline scripts
COPY backend/config.py backend/
COPY backend/pipeline/__init__.py backend/pipeline/
COPY scripts/download_model.py scripts/

# Pre-download the quantized ONNX model during build time
RUN python scripts/download_model.py

# Copy the rest of the application
COPY . .

# Hugging Face Spaces runs on port 7860 by default
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
