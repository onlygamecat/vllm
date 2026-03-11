# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.datastructures import State
from starlette.types import ASGIApp, Receive, Scope, Send

SCALING_EXEMPT_PATHS = frozenset(
    {
        "/health",
        "/scale_data_parallel",
        "/scale_elastic_ep",
        "/is_scaling_data_parallel",
        "/is_scaling_elastic_ep",
    }
)


def initialize_scaling_state(state: State) -> None:
    if not hasattr(state, "is_scaling_data_parallel"):
        state.is_scaling_data_parallel = False
    if not hasattr(state, "scaling_data_parallel_lock"):
        state.scaling_data_parallel_lock = threading.Lock()
    if not hasattr(state, "scaling_data_parallel_exempt_paths"):
        state.scaling_data_parallel_exempt_paths = SCALING_EXEMPT_PATHS


def is_scaling_data_parallel(state: State) -> bool:
    initialize_scaling_state(state)
    return bool(state.is_scaling_data_parallel)


def set_scaling_data_parallel(state: State, value: bool) -> None:
    initialize_scaling_state(state)
    state.is_scaling_data_parallel = value


def get_scaling_data_parallel_lock(state: State) -> threading.Lock:
    initialize_scaling_state(state)
    return state.scaling_data_parallel_lock


class ScalingMiddleware:
    """Return 503 for ordinary requests while DP scaling is in progress."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        state = scope["app"].state
        initialize_scaling_state(state)
        path = scope.get("path", "")
        exempt_paths = state.scaling_data_parallel_exempt_paths
        if path in exempt_paths or path.startswith("/metrics"):
            return self.app(scope, receive, send)

        if state.is_scaling_data_parallel:
            response = JSONResponse(
                content={
                    "error": "The model is currently scaling. Please try again later."
                },
                status_code=503,
            )
            return response(scope, receive, send)

        return self.app(scope, receive, send)
