# Use a slim Python base image
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc pkg-config python3-dev libmupdf-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements inline for offline build
RUN pip install --no-cache-dir pymupdf numpy

# Copy main script
COPY main.py /app/main.py

# Create input/output folders (for local dev, Docker run will mount these)
RUN mkdir -p /app/input /app/output

ENTRYPOINT ["python", "/app/main.py"] 