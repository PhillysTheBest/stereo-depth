FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra cpu --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra cpu
ENV PATH="/app/.venv/bin:$PATH" STEREO_DATA_DIR=/app/data STEREO_PRELOAD_ENGINES=true
# Bake the RAFT-Stereo weights in when the download works; otherwise the container retries at
# start. Either way SGBM serves. World-writable so Hugging Face Spaces' non-root user can write.
RUN mkdir -p /app/data && (python -m stereo_depth.weights || true) && chmod -R a+rwX /app/data
EXPOSE 7860
CMD ["sh", "-c", "python -m stereo_depth.weights; exec uvicorn stereo_depth.api.app:app --host 0.0.0.0 --port 7860"]
