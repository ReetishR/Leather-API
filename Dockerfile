# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System packages needed by TensorFlow/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . /app

# Env defaults (override in AWS)
ENV PORT=5000 \
    FLASK_ENV=production

EXPOSE 5000

# Start via Gunicorn
CMD ["uvicorn", "simple_api:app", "--host", "0.0.0.0", "--port", "5000"]
