# Use multi-stage build with slim Debian image
FROM python:3.8-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.0+cpu torchvision==0.19.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir opencv-python-headless==4.10.0.82 && \
    pip install --no-cache-dir Flask==3.0.3 numpy==1.24.4 matplotlib==3.7.5 tqdm redis && \
    pip cache purge

# Production stage
FROM python:3.8-slim-bullseye

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code (exclude unnecessary files)
COPY --chown=appuser:appgroup app.py requirements.txt ./
COPY --chown=appuser:appgroup *.py ./
COPY --chown=appuser:appgroup databases/ ./databases/
COPY --chown=appuser:appgroup handler/ ./handler/
COPY --chown=appuser:appgroup model/ ./model/
COPY --chown=appuser:appgroup model_data/ ./model_data/
COPY --chown=appuser:appgroup nets/ ./nets/
COPY --chown=appuser:appgroup nets_retinaface/ ./nets_retinaface/
COPY --chown=appuser:appgroup utils/ ./utils/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Run the application
CMD ["python", "app.py"]