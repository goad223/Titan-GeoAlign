# =============================================================================
#  Titan-GeoAlign — CPU Dockerfile
#  Multi-stage build for production deployment (no GPU required).
# =============================================================================

# ── Stage 1: build wheels ─────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      g++ \
      git \
      curl \
      libgdal-dev \
      gdal-bin \
      libproj-dev \
      libgeos-dev \
      libspatialindex-dev \
      libhdf5-dev \
      libnetcdf-dev \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

WORKDIR /build

# Copy dependency files first (layer cache optimisation)
COPY requirements.txt ./
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Titan-GeoAlign"
LABEL org.opencontainers.image.description="Sub-pixel geospatial image alignment pipeline"
LABEL org.opencontainers.image.source="https://github.com/goad223/Titan-GeoAlign"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ARG DEBIAN_FRONTEND=noninteractive
ARG APP_USER=titan
ARG APP_UID=1000

# Minimal runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgdal32 \
      libproj25 \
      libgeos3.11.1 \
      libspatialindex7 \
      libhdf5-103-1 \
      libnetcdf19 \
      curl \
      tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root application user
RUN useradd --uid "${APP_UID}" --create-home --shell /bin/bash "${APP_USER}"

# Install pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

WORKDIR /app

# Copy application source
COPY --chown=${APP_USER}:${APP_USER} . .

# Install package in editable mode (no network access needed for source)
RUN pip install --no-deps --no-build-isolation -e . 2>/dev/null || true

# Create runtime directories
RUN mkdir -p data/dem_cache models/foundation models/matchers logs \
    && chown -R "${APP_USER}:${APP_USER}" /app

USER ${APP_USER}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "gui.backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
