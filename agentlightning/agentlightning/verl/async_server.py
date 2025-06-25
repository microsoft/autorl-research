from typing import Type, TypeVar

import ray
import verl.workers.rollout.async_server
from agentlightning.instrumentation.vllm import instrument_vllm
from verl.workers.rollout.async_server import AsyncServerBase
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer


_original_async_server_class = None


def instrumented_async_server_class(rollout_backend: str) -> Type[AsyncServerBase]:
    if rollout_backend == "vllm_agentlightning":
        return InstrumentedAsyncvLLMServer

    else:
        return _original_async_server_class(rollout_backend)  # type: ignore


def instrument_async_server():
    global _original_async_server_class
    _original_async_server_class = verl.workers.rollout.async_server.async_server_class
    verl.workers.rollout.async_server.async_server_class = instrumented_async_server_class


T = TypeVar("T", bound=AsyncServerBase)


def _unwrap_ray_remote(cls: T) -> T:
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


@ray.remote(num_cpus=1)
class InstrumentedAsyncvLLMServer(_unwrap_ray_remote(AsyncvLLMServer)):  # type: ignore
    def __init__(self, *args, **kwargs):
        instrument_vllm()
        super().__init__(*args, **kwargs)
