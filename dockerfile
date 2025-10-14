# Dockerfile

# 1. Use a lightweight official Python image
FROM python:3.10-slim

# 2. Prevent Python from writing .pyc files and enable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Install system dependencies (if needed for aioredis, ccxt, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Set working directory
WORKDIR /app

# 5. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Expose ports for Prometheus metrics (modify as needed)
EXPOSE 8000 8001 8002

# 8. Default command (can be overridden by docker-compose)
CMD ["python", "train_model.py"]
