# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from vllm.entrypoints.serve.elastic_ep.api_router import (
    scale_data_parallel,
    scale_elastic_ep,
)
from vllm.entrypoints.serve.elastic_ep.middleware import (
    ScalingMiddleware,
    get_scaling_data_parallel_lock,
    initialize_scaling_state,
)


def make_mock_request(
    *,
    data_parallel_backend: str = "ray",
    data_parallel_size: int = 2,
    data_parallel_external_lb: bool = False,
    data_parallel_hybrid_lb: bool = False,
    api_server_count: int = 1,
    headless: bool = False,
    scale_side_effect: BaseException | None = None,
) -> tuple[Mock, AsyncMock]:
    request = Mock(spec=Request)
    app_state = SimpleNamespace()
    initialize_scaling_state(app_state)

    parallel_config = SimpleNamespace(
        data_parallel_backend=data_parallel_backend,
        data_parallel_size=data_parallel_size,
        data_parallel_external_lb=data_parallel_external_lb,
        data_parallel_hybrid_lb=data_parallel_hybrid_lb,
    )
    engine_client = AsyncMock()
    engine_client.vllm_config = SimpleNamespace(parallel_config=parallel_config)
    engine_client.scale_data_parallel = AsyncMock(side_effect=scale_side_effect)

    app_state.engine_client = engine_client
    app_state.args = SimpleNamespace(
        api_server_count=api_server_count,
        headless=headless,
    )
    request.app.state = app_state
    request.json = AsyncMock(
        return_value={"new_data_parallel_size": 3, "drain_timeout": 60}
    )
    return request, engine_client


@pytest.mark.asyncio
async def test_scale_data_parallel_success():
    request, engine_client = make_mock_request()

    response = await scale_data_parallel(request)

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "message": "Scaled to 3 data parallel engines",
        "new_data_parallel_size": 3,
    }
    engine_client.scale_data_parallel.assert_awaited_once_with(3, 60)
    assert request.app.state.is_scaling_data_parallel is False


@pytest.mark.asyncio
async def test_scale_elastic_ep_alias_calls_scale_data_parallel():
    request, engine_client = make_mock_request()

    response = await scale_elastic_ep(request)

    assert response.status_code == 200
    engine_client.scale_data_parallel.assert_awaited_once_with(3, 60)


@pytest.mark.asyncio
async def test_scale_data_parallel_rejects_invalid_deployment():
    request, engine_client = make_mock_request(data_parallel_backend="mp")

    with pytest.raises(HTTPException) as exc_info:
        await scale_data_parallel(request)

    assert exc_info.value.status_code == 400
    engine_client.scale_data_parallel.assert_not_called()


@pytest.mark.asyncio
async def test_scale_data_parallel_returns_conflict_when_already_scaling():
    request, engine_client = make_mock_request()
    scaling_lock = get_scaling_data_parallel_lock(request.app.state)
    assert scaling_lock.acquire(blocking=False)

    try:
        with pytest.raises(HTTPException) as exc_info:
            await scale_data_parallel(request)
    finally:
        scaling_lock.release()

    assert exc_info.value.status_code == 409
    engine_client.scale_data_parallel.assert_not_called()


@pytest.mark.asyncio
async def test_scale_data_parallel_returns_timeout_and_clears_state():
    request, engine_client = make_mock_request(scale_side_effect=TimeoutError())

    with pytest.raises(HTTPException) as exc_info:
        await scale_data_parallel(request)

    assert exc_info.value.status_code == 408
    engine_client.scale_data_parallel.assert_awaited_once_with(3, 60)
    assert request.app.state.is_scaling_data_parallel is False


def test_scaling_middleware_returns_503_for_regular_requests():
    app = FastAPI()
    initialize_scaling_state(app.state)
    app.add_middleware(ScalingMiddleware)

    @app.get("/v1/completions")
    async def completions():
        return {"ok": True}

    @app.post("/scale_data_parallel")
    async def scale():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    app.state.is_scaling_data_parallel = True

    with TestClient(app) as client:
        response = client.get("/v1/completions")
        assert response.status_code == 503
        assert response.json()["error"] == (
            "The model is currently scaling. Please try again later."
        )

        scale_response = client.post("/scale_data_parallel")
        assert scale_response.status_code == 200

        health_response = client.get("/health")
        assert health_response.status_code == 200
