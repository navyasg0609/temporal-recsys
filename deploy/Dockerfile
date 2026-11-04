# The recommendation service. The image holds code only: the bundle is built from rating
# data, so it is mounted at run time and never copied in (decision 038).
# Build from the repository root: docker build -f deploy/Dockerfile -t temporal-recsys:dev .

FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so code changes do not reinstall them. The service imports neither
# pyarrow nor scipy, and a test enforces that, so they stay out of the image.
ARG SKIP="--no-install-package pyarrow --no-install-package scipy"
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --extra serve $SKIP --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --extra serve $SKIP --no-editable

FROM python:3.13-slim
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    BUNDLE_PATH=/data/bundle.npz \
    PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8000
CMD ["uvicorn", "--factory", "temporal_recsys.serving.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
