services:
  redis:
    image: redis:7-alpine
    container_name: zazi-redis
    command: redis-server --appendonly yes
    ports:
      - "6380:6379"  # Using 6380 on host to avoid conflict with local Redis
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - zazi-network

  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: zazi-backend
    env_file:
      - .env
    environment:
      ENVIRONMENT: production
      REDIS_URL: redis://redis:6379/0
      BACKEND_URL: http://204.236.233.91:8000
    ports:
      - "8000:8000"
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - zazi-network

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: zazi-worker
    command: python -m tasks.cli
    env_file:
      - .env
    environment:
      ENVIRONMENT: production
      REDIS_URL: redis://redis:6379/0
      BACKEND_URL: http://204.236.233.91:8000
    deploy:
      resources:
        limits:
          memory: 4G  # Reduced for 8GB instance (will still likely fail - upgrade recommended)
        reservations:
          memory: 1G
    depends_on:
      redis:
        condition: service_healthy
      backend:
        condition: service_started
    restart: unless-stopped
    networks:
      - zazi-network

  frontend:
    build:
      context: ../frontend
      dockerfile: Dockerfile
      args:
        NEXT_PUBLIC_API_URL: http://204.236.233.91:8000
        NEXT_PUBLIC_APP_URL: http://204.236.233.91:3001
    container_name: zazi-frontend
    environment:
      NODE_ENV: production
      NEXT_PUBLIC_API_URL: http://204.236.233.91:8000
      NEXT_PUBLIC_APP_URL: http://204.236.233.91:3001
    ports:
      - "3001:3000"
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - zazi-network

  website:
    build:
      context: ../website
      dockerfile: Dockerfile
      args:
        NEXT_PUBLIC_API_URL: http://204.236.233.91:8000
        NEXT_PUBLIC_APP_URL: http://204.236.233.91:3000
        NEXT_PUBLIC_FRONTEND_URL: http://204.236.233.91:3001
    container_name: zazi-website
    environment:
      NODE_ENV: production
      NEXT_PUBLIC_API_URL: http://204.236.233.91:8000
      NEXT_PUBLIC_APP_URL: http://204.236.233.91:3000
      NEXT_PUBLIC_FRONTEND_URL: http://204.236.233.91:3001
    ports:
      - "3000:3000"
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - zazi-network

volumes:
  redis_data:
    driver: local

networks:
  zazi-network:
    driver: bridge


services:
  redis:
    image: redis:7-alpine
    container_name: zazi-redis
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"  # Standard Redis port for production
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - zazi-network
    restart: unless-stopped

  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: zazi-backend
    env_file:
      - .env
    environment:
      ENVIRONMENT: production
      REDIS_URL: redis://redis:6379/0
      BACKEND_URL: http://204.236.233.91:8000
    ports:
      - "8000:8000"
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped
    networks:
      - zazi-network

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: zazi-worker
    command: python -m tasks.cli
    env_file:
      - .env
    environment:
      ENVIRONMENT: production
      REDIS_URL: redis://redis:6379/0
      BACKEND_URL: http://204.236.233.91:8000
    # Memory limits for Demucs (requires significant RAM)
    # Current instance: c7i-flex.large (8GB RAM total)
    # WARNING: 8GB is NOT enough for Demucs - upgrade to c7i-flex.xlarge (16GB) or larger
    # Demucs typically needs 4-8GB RAM per file + OS/Docker overhead
    deploy:
      resources:
        limits:
          memory: 4G  # Reduced for 8GB instance (will still likely fail - upgrade recommended)
        reservations:
          memory: 1G
    depends_on:
      redis:
        condition: service_healthy
      backend:
        condition: service_started
    restart: unless-stopped
    networks:
      - zazi-network

  frontend:
    build:
      context: ../frontend
      dockerfile: Dockerfile
      args:
        NEXT_PUBLIC_API_URL: http://204.236.233.91:8000
        NEXT_PUBLIC_APP_URL: http://204.236.233.91:3001
    container_name: zazi-frontend
    environment:
      NODE_ENV: production
      NEXT_PUBLIC_API_URL: http://204.236.233.91:8000
      NEXT_PUBLIC_APP_URL: http://204.236.233.91:3001
    ports:
      - "3001:3000"
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - zazi-network

  website:
    build:
      context: ../website
      dockerfile: Dockerfile
      args:
        NEXT_PUBLIC_API_URL: http://204.236.233.91:8000
        NEXT_PUBLIC_APP_URL: http://204.236.233.91:3000
        NEXT_PUBLIC_FRONTEND_URL: http://204.236.233.91:3001
    container_name: zazi-website
    environment:
      NODE_ENV: production
      NEXT_PUBLIC_API_URL: http://204.236.233.91:8000
      NEXT_PUBLIC_APP_URL: http://204.236.233.91:3000
      NEXT_PUBLIC_FRONTEND_URL: http://204.236.233.91:3001
    ports:
      - "3000:3000"
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - zazi-network

volumes:
  redis_data:
    driver: local

networks:
  zazi-network:
    driver: bridge



FROM python:3.11-slim AS builder

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    libpq-dev \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# ================================
# Runtime stage (production)
# ================================
FROM python:3.11-slim AS runtime

# Install only runtime system dependencies (including libgomp1 for torchaudio and ffmpeg for demucs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos "" --uid 5678 appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Copy application code (excluding files in .dockerignore)
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Start application with gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "20", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--worker-tmp-dir", "/dev/shm", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "50", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "main:create_app"]