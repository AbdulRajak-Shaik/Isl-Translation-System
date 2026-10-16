# Stage 1: Build the React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend Runtime
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend, ML code, manifests, and weights
COPY backend/ backend/
COPY ml/ ml/
COPY data/manifests/ data/manifests/
COPY vocab.json ./
COPY isl_transformer_best.pt ./

# Copy compiled frontend from Stage 1 into the Python app
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONPATH=/app
ENV MODEL_DEVICE=cpu

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
