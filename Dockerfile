# Multi-stage build. Dependencies are installed into a virtualenv in the builder
# and copied into a slim runtime, so build toolchains never ship to production.

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied before the source so a code change does not invalidate the dependency
# layer.
COPY requirements.txt .
RUN pip install -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Run as a non-root user: a container that can write to its own code directory
# turns a file-write bug into remote code execution.
RUN useradd --create-home --uid 1000 docubot \
    && mkdir -p /app/data/vector_db /app/data/images /app/logs \
    && chown -R docubot:docubot /app

COPY --chown=docubot:docubot . .

USER docubot

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/v1/health').status==200 else 1)"

CMD ["uvicorn", "api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
