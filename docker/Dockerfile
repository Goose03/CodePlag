# ---------- builder ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url ${TORCH_INDEX} \
    && pip install .

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" HF_HOME=/app/.cache/huggingface
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
RUN mkdir -p /app/data /app/models /app/.cache && chown -R appuser:appuser /app
USER appuser
ENTRYPOINT ["codeplag"]
CMD ["--help"]
