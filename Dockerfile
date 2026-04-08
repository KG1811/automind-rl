# Use Python 3.11 so dependency wheels resolve cleanly on HF/Render
FROM python:3.11-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for many Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better caching)
COPY requirements.txt .

# Upgrade pip tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose port
EXPOSE 7860

# Start server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
