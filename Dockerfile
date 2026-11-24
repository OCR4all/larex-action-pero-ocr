FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

ARG FLAVOR=cpu

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_CACHE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --extra "${FLAVOR}"

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra "${FLAVOR}"

FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PERO_MODEL_DIR=/models
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

WORKDIR /app

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin larex-action \
    && install -d -o 10001 -g 10001 /models

COPY --from=builder /opt/venv /opt/venv

USER 10001:10001
EXPOSE 9000
VOLUME /models
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9000/health', timeout=4)"]
CMD ["uvicorn", "larex_action:app", "--host", "0.0.0.0", "--port", "9000", "--timeout-graceful-shutdown", "5"]
