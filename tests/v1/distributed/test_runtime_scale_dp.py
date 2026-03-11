# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import cast

import pytest
import torch

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
BASE_DP_SIZE = int(os.getenv("DP_SIZE", "2"))
SCALE_UP_DP_SIZE = int(os.getenv("SCALE_UP_DP_SIZE", str(BASE_DP_SIZE + 1)))


def make_stats_logger(stats_loggers: dict[int, object]):
    @dataclass
    class CountingStatsLogger(StatLoggerBase):
        init_count: int = 0
        finished_req_count: int = 0

        def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
            del vllm_config
            stats_loggers[engine_index] = self

        def record(
            self,
            scheduler_stats: SchedulerStats | None,
            iteration_stats: IterationStats | None,
            mm_cache_stats: MultiModalCacheStats | None = None,
            engine_idx: int = 0,
        ):
            del scheduler_stats, mm_cache_stats, engine_idx
            if iteration_stats:
                self.finished_req_count += len(iteration_stats.finished_requests)

        def log_engine_initialized(self):
            self.init_count += 1

    return CountingStatsLogger


async def generate_requests(
    engine: AsyncLLM, prefix: str, num_requests: int, max_tokens: int = 8
) -> None:
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    tasks = []
    for idx in range(num_requests):
        request_id = f"{prefix}-{idx}"
        tasks.append(
            asyncio.create_task(
                _run_generation(engine, request_id, sampling_params=sampling_params)
            )
        )
        await asyncio.sleep(0.01)
    await asyncio.gather(*tasks)


async def _run_generation(
    engine: AsyncLLM, request_id: str, sampling_params: SamplingParams
) -> None:
    final_output = None
    async for output in engine.generate(
        request_id=request_id,
        prompt="Runtime DP scaling test prompt.",
        sampling_params=sampling_params,
    ):
        final_output = output
    assert final_output is not None


async def wait_for_engines_to_idle(engine: AsyncLLM) -> None:
    core_client = cast(DPLBAsyncMPClient, engine.engine_core)
    for _ in range(20):
        if not core_client.engines_running and not core_client.reqs_in_flight:
            return
        await asyncio.sleep(0.5)
    raise AssertionError("DP engines did not become idle in time.")


def make_engine_args(dp_size: int) -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model=MODEL_NAME,
        enforce_eager=True,
        dtype="auto",
        max_model_len=256,
        max_num_seqs=32,
        tensor_parallel_size=1,
        data_parallel_size=dp_size,
        data_parallel_backend="ray",
        disable_custom_all_reduce=True,
    )


def skip_if_ray_dp_unavailable():
    if current_platform.is_rocm():
        pytest.skip("Ray DP backend is not supported on ROCm.")
    if not torch.cuda.is_available():
        pytest.skip("Runtime DP scaling tests require CUDA.")


@pytest.mark.asyncio
async def test_runtime_scale_dense_dp_down_to_one_and_back_to_two():
    skip_if_ray_dp_unavailable()
    if BASE_DP_SIZE < 2:
        pytest.skip("This test requires at least two initial DP ranks.")

    stats_loggers: dict[int, object] = {}
    CountingStatsLogger = make_stats_logger(stats_loggers)

    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(
            make_engine_args(BASE_DP_SIZE), stat_loggers=[CountingStatsLogger]
        )
        after.callback(engine.shutdown)

        await generate_requests(engine, "before-scale-down", num_requests=24)
        await wait_for_engines_to_idle(engine)

        core_client = cast(DPLBAsyncMPClient, engine.engine_core)
        assert len(core_client.core_engines) == BASE_DP_SIZE
        assert len(core_client.lb_engines) == BASE_DP_SIZE

        removed_rank = BASE_DP_SIZE - 1
        removed_rank_finished = stats_loggers[removed_rank].finished_req_count

        await engine.scale_data_parallel(1)
        await wait_for_engines_to_idle(engine)

        assert core_client.engine_ranks_managed == [0]
        assert len(core_client.core_engines) == 1
        assert len(core_client.lb_engines) == 1
        assert not core_client.reqs_in_flight

        await generate_requests(engine, "single-rank", num_requests=12)
        await wait_for_engines_to_idle(engine)
        assert stats_loggers[removed_rank].finished_req_count == removed_rank_finished

        await engine.scale_data_parallel(BASE_DP_SIZE)
        await wait_for_engines_to_idle(engine)

        assert core_client.engine_ranks_managed == list(range(BASE_DP_SIZE))
        assert len(core_client.core_engines) == BASE_DP_SIZE
        assert len(core_client.lb_engines) == BASE_DP_SIZE

        await generate_requests(engine, "scale-back-up", num_requests=24)
        await wait_for_engines_to_idle(engine)
        assert stats_loggers[removed_rank].finished_req_count > 0


@pytest.mark.asyncio
async def test_runtime_scale_dense_dp_up_adds_new_rank():
    skip_if_ray_dp_unavailable()
    if BASE_DP_SIZE < 2:
        pytest.skip("This test requires at least two initial DP ranks.")
    if torch.cuda.device_count() < SCALE_UP_DP_SIZE:
        pytest.skip(
            f"Need at least {SCALE_UP_DP_SIZE} visible CUDA devices for scale-up."
        )

    stats_loggers: dict[int, object] = {}
    CountingStatsLogger = make_stats_logger(stats_loggers)

    with ExitStack() as after:
        engine = AsyncLLM.from_engine_args(
            make_engine_args(BASE_DP_SIZE), stat_loggers=[CountingStatsLogger]
        )
        after.callback(engine.shutdown)

        await generate_requests(engine, "before-scale-up", num_requests=24)
        await wait_for_engines_to_idle(engine)

        core_client = cast(DPLBAsyncMPClient, engine.engine_core)
        await engine.scale_data_parallel(SCALE_UP_DP_SIZE)
        await wait_for_engines_to_idle(engine)

        assert core_client.engine_ranks_managed == list(range(SCALE_UP_DP_SIZE))
        assert len(core_client.core_engines) == SCALE_UP_DP_SIZE
        assert len(core_client.lb_engines) == SCALE_UP_DP_SIZE

        await generate_requests(engine, "after-scale-up", num_requests=48)
        await wait_for_engines_to_idle(engine)

        new_rank = SCALE_UP_DP_SIZE - 1
        assert new_rank in stats_loggers
        assert stats_loggers[new_rank].finished_req_count > 0
        assert not core_client.reqs_in_flight
