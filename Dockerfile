# Geo-Financial Intelligence Platform - Docker Container
# ====================================================

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models output logs

# Expose port for API (future use)
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app/src

# Default command - run the main demo
CMD ["python", "main.py", "--demo"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.append('src'); from feature_engineering.hexgrid import create_porto_alegre_grid; print('✅ Container healthy')" || exit 1

# Labels for metadata
LABEL maintainer="Geo-Financial Intelligence Platform"
LABEL version="1.0"
LABEL description="Advanced geospatial data science platform for financial technology applications"