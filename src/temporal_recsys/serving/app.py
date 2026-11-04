"""The recommendation web service.

Run with `uvicorn --factory temporal_recsys.serving.app:create_app`. The bundle loads
before the server accepts connections, so a reachable `/healthz` means it can serve.
Logs are one JSON object per line and leave out user IDs.
"""

import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as PathParam
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from temporal_recsys.serving.bundle import load_bundle
from temporal_recsys.serving.recommender import Recommender, Source

BUNDLE_PATH_VARIABLE = "BUNDLE_PATH"
DEFAULT_BUNDLE_PATH = Path("/data/bundle.npz")
MAX_K = 100

LOGGER = logging.getLogger("temporal_recsys.serving")


class RecommendationResponse(BaseModel):
    user_id: int
    source: Source
    film_ids: list[int]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = {"time": self.formatTime(record), "level": record.levelname}
        fields |= getattr(record, "fields", {"message": record.getMessage()})
        return json.dumps(fields)


def create_app(bundle_path: Path | None = None) -> FastAPI:
    path = bundle_path or Path(os.environ.get(BUNDLE_PATH_VARIABLE, DEFAULT_BUNDLE_PATH))
    bundle = load_bundle(path)
    recommender = Recommender(bundle)
    _configure_logging()
    LOGGER.info(
        "bundle loaded",
        extra={"fields": {"event": "bundle_loaded", "users": len(bundle.user_ids)}},
    )

    registry = CollectorRegistry()
    requests = Counter(
        "requests", "HTTP requests by route and status", ["route", "status"], registry=registry
    )
    latency = Histogram(
        "request_seconds",
        "Time to answer a request",
        ["route"],
        buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        registry=registry,
    )
    sources = Counter(
        "recommendations", "Recommendation lists by source", ["source"], registry=registry
    )

    app = FastAPI(title="temporal-recsys")

    @app.middleware("http")
    async def measure(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        seconds = time.perf_counter() - start
        # The route template, such as /recommendations/{user_id}, keeps user IDs out.
        route = getattr(request.scope.get("route"), "path", "unmatched")
        requests.labels(route, str(response.status_code)).inc()
        latency.labels(route).observe(seconds)
        LOGGER.info(
            "request",
            extra={
                "fields": {
                    "event": "request",
                    "method": request.method,
                    "route": route,
                    "status": response.status_code,
                    "milliseconds": round(seconds * 1000, 3),
                }
            },
        )
        return response

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/recommendations/{user_id}")
    def recommend(
        user_id: Annotated[int, PathParam(ge=1)],
        k: Annotated[int, Query(ge=1, le=MAX_K)] = 10,
    ) -> RecommendationResponse:
        if k > bundle.n_films:
            raise HTTPException(422, f"k must be at most {bundle.n_films}, the catalog size")
        recommendation = recommender.recommend(user_id, k)
        sources.labels(recommendation.source.value).inc()
        return RecommendationResponse(
            user_id=recommendation.user_id,
            source=recommendation.source,
            film_ids=recommendation.film_ids,
        )

    return app


def _configure_logging() -> None:
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)
        LOGGER.propagate = False
