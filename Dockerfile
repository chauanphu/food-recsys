FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY src/ src/
COPY main.py .
# Copy data files just in case
COPY *.csv .
COPY *.json .

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Keep stdout/stderr unbuffered
ENV PYTHONUNBUFFERED=1

# Create temp directory for uploads
RUN mkdir -p /tmp/food-recsys/uploads

# Expose ports
EXPOSE 8000
EXPOSE 8501
