# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.elastic_ep.middleware import (
    get_scaling_data_parallel_lock,
    is_scaling_data_parallel,
    set_scaling_data_parallel,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def _validate_runtime_dp_scaling_support(raw_request: Request) -> None:
    client = engine_client(raw_request)
    parallel_config = client.vllm_config.parallel_config
    args = raw_request.app.state.args

    if parallel_config.data_parallel_backend != "ray":
        raise HTTPException(
            status_code=400,
            detail="Runtime data parallel scaling only supports "
            "data_parallel_backend=ray.",
        )

    if (
        parallel_config.data_parallel_external_lb
        or parallel_config.data_parallel_hybrid_lb
    ):
        raise HTTPException(
            status_code=400,
            detail="Runtime data parallel scaling only supports internal "
            "load balancing deployments.",
        )

    if getattr(args, "headless", False):
        raise HTTPException(
            status_code=400,
            detail="Runtime data parallel scaling is not supported in headless mode.",
        )

    if getattr(args, "api_server_count", 1) != 1:
        raise HTTPException(
            status_code=400,
            detail="Runtime data parallel scaling currently requires "
            "api_server_count=1.",
        )


def _validate_scale_request(
    raw_request: Request, new_data_parallel_size: int, drain_timeout: int
) -> None:
    _validate_runtime_dp_scaling_support(raw_request)

    if not isinstance(new_data_parallel_size, int) or new_data_parallel_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="new_data_parallel_size must be a positive integer",
        )

    current_size = (
        engine_client(raw_request).vllm_config.parallel_config.data_parallel_size
    )
    if new_data_parallel_size == current_size:
        raise HTTPException(
            status_code=400,
            detail="new_data_parallel_size must differ from the current "
            f"data parallel size {current_size}.",
        )

    if not isinstance(drain_timeout, int) or drain_timeout <= 0:
        raise HTTPException(
            status_code=400, detail="drain_timeout must be a positive integer"
        )


router = APIRouter()


async def _scale_data_parallel_impl(raw_request: Request) -> JSONResponse:
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e

    new_data_parallel_size = body.get("new_data_parallel_size")
    drain_timeout = body.get("drain_timeout", 120)

    if new_data_parallel_size is None:
        raise HTTPException(
            status_code=400, detail="new_data_parallel_size is required"
        )

    _validate_scale_request(raw_request, new_data_parallel_size, drain_timeout)

    scaling_lock = get_scaling_data_parallel_lock(raw_request.app.state)
    if not scaling_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A data parallel scaling operation is already in progress.",
        )

    client = engine_client(raw_request)
    set_scaling_data_parallel(raw_request.app.state, True)
    try:
        await client.scale_data_parallel(new_data_parallel_size, drain_timeout)
        return JSONResponse(
            {
                "message": (
                    f"Scaled to {new_data_parallel_size} data parallel engines"
                ),
                "new_data_parallel_size": new_data_parallel_size,
            }
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail="Scale failed due to request drain timeout "
            f"after {drain_timeout} seconds",
        ) from e
    except (NotImplementedError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Scale failed: %s", e)
        raise HTTPException(status_code=500, detail="Scale failed") from e
    finally:
        set_scaling_data_parallel(raw_request.app.state, False)
        scaling_lock.release()


@router.post(
    "/scale_data_parallel",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.CONFLICT.value: {"model": ErrorResponse},
        HTTPStatus.REQUEST_TIMEOUT.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def scale_data_parallel(raw_request: Request):
    return await _scale_data_parallel_impl(raw_request)


@router.post(
    "/scale_elastic_ep",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.CONFLICT.value: {"model": ErrorResponse},
        HTTPStatus.REQUEST_TIMEOUT.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def scale_elastic_ep(raw_request: Request):
    return await _scale_data_parallel_impl(raw_request)


@router.post("/is_scaling_data_parallel")
async def is_scaling_dp(raw_request: Request):
    return JSONResponse(
        {"is_scaling_data_parallel": is_scaling_data_parallel(raw_request.app.state)}
    )


@router.post("/is_scaling_elastic_ep")
async def is_scaling_elastic_ep(raw_request: Request):
    return JSONResponse(
        {"is_scaling_elastic_ep": is_scaling_data_parallel(raw_request.app.state)}
    )


def attach_router(app: FastAPI):
    app.include_router(router)
