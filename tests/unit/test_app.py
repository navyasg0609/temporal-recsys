import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from temporal_recsys.models.factors import FactorModel
from temporal_recsys.serving.app import LOGGER, JsonFormatter, create_app
from temporal_recsys.serving.bundle import build_bundle, save_bundle


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    factors = FactorModel(
        user_ids=np.array([10, 20], dtype=np.int32),
        user_factors=np.array([[1.0, 0.0], [0.0, 1.0]]),
        film_factors=np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.0, 0.0]]),
    )
    bundle = build_bundle(
        factors,
        np.array([2, 2, 3], dtype=np.int32),
        np.array([20, 10, 30, 10], dtype=np.int32),
        np.array([2, 3, 1, 1], dtype=np.int32),
        4,
    )
    save_bundle(bundle, tmp_path / "bundle.npz")
    return TestClient(create_app(tmp_path / "bundle.npz"))


def test_known_user_gets_the_models_list(client: TestClient) -> None:
    response = client.get("/recommendations/10", params={"k": 2})

    assert response.status_code == 200
    assert response.json() == {"user_id": 10, "source": "model", "film_ids": [2, 4]}


def test_unknown_user_gets_popular_films(client: TestClient) -> None:
    response = client.get("/recommendations/99", params={"k": 2})

    assert response.json() == {"user_id": 99, "source": "popular", "film_ids": [2, 3]}


@pytest.mark.parametrize(
    "url", ["/recommendations/0", "/recommendations/x", "/recommendations/10?k=0"]
)
def test_bad_input_is_rejected(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 422


def test_k_beyond_the_catalog_is_rejected(client: TestClient) -> None:
    # k passes the request check but exceeds the 4 films in this bundle.
    response = client.get("/recommendations/10", params={"k": 5})

    assert response.status_code == 422
    assert "at most 4" in response.json()["detail"]


def test_health_and_metrics(client: TestClient) -> None:
    client.get("/recommendations/10", params={"k": 2})
    client.get("/recommendations/99", params={"k": 2})

    assert client.get("/healthz").json() == {"status": "ok"}
    metrics = client.get("/metrics").text
    assert 'recommendations_total{source="model"} 1.0' in metrics
    assert 'requests_total{route="/recommendations/{user_id}",status="200"} 2.0' in metrics


def test_request_logs_are_json_without_user_ids(client: TestClient) -> None:
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())
    LOGGER.addHandler(handler)
    try:
        client.get("/recommendations/12345", params={"k": 2})
    finally:
        LOGGER.removeHandler(handler)

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    request = next(line for line in lines if line.get("event") == "request")
    assert request["route"] == "/recommendations/{user_id}"
    assert "12345" not in json.dumps(lines)


def test_serving_imports_only_what_the_image_installs() -> None:
    code = (
        "import sys, temporal_recsys.serving.app; "
        "print(sorted({'pyspark', 'torch', 'pyarrow', 'scipy'} & set(sys.modules)))"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
