# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Exercise real executor startup through drafter wiring, before device setup."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from dataclasses import dataclass
from test.runtime.cache_pool_test_utils import MinimalCacheView, make_arena, one_group

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution import model_executor
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.plugins import registry


@dataclass
class _LoadedModel:
    model: torch.nn.Module
    mapping: Mapping
    model_config: PretrainedConfig


class _CacheConsumer:
    cache_consumer_attn_kinds = ("full_history",)


class _TargetWired(Exception):
    """Stop the actual constructor at the documented pre-device boundary."""


class _Drafter(BaseDrafter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.received_config = None
        self.configure_calls = 0

    def configure_algorithm(self, algorithm_config):
        self.received_config = dict(algorithm_config)
        self.configure_calls += 1

    def wire_target(self, target):
        raise _TargetWired(self, target)

    def run(self, *args, **kwargs):
        raise AssertionError("startup test must not execute a draft round")

    def draft(self, *args, **kwargs):
        raise AssertionError("startup test must not draft tokens")


@pytest.mark.parametrize("algorithm_config", [None, {}, {"depth": 6}])
def test_executor_configures_plugin_before_wiring_target(algorithm_config):
    config = model_executor.ModelExecutorConfig(
        max_req_pool_size=2,
        output_length=2,
        enforce_eager=True,
        prefix_granularity=1,
        max_num_seqs=1,
        chunked_prefill_size=2,
        vocab_size=8,
        context_len=8,
        physical_context_len=14,
        device="cpu",
        gpu_id=0,
        global_rank=0,
        cudagraph_capture_sizes=None,
        disable_cuda_graph_padding=True,
        spec_topk=1,
        max_cudagraph_capture_size=1,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=None,
        input_logprob_chunk_tokens=1,
        enable_speculative_sampling=False,
        query_shard_size=1,
        query_shard_rank=0,
        enable_memory_saver=False,
        spec_algo="STARTUP_FIXTURE",
        spec_num_steps=1,
        spec_num_tokens=2,
        spec_algorithm_config=algorithm_config,
    )
    plan = pack(
        (one_group("history", CacheFieldSpec("history.k", "kv", (1,), "uint8")),),
        prefix_granularity=1,
    ).bind(2)
    pool = MinimalCacheView(make_arena(plan, device="cpu"), torch.uint8, rank=0)
    model = _LoadedModel(
        torch.nn.Identity(),
        Mapping(rank=0),
        PretrainedConfig(
            hf_text_config=PretrainedConfig(), requires_request_token_history=False
        ),
    )
    # The sentinel unwinds registration and stops before sampling/stream/graph setup.
    # Real InputBuffers, RuntimeStates and the cache-contract validator run first.
    with pytest.raises(_TargetWired) as stopped:
        with registry.recording():
            registry.register_drafter("STARTUP_FIXTURE", _Drafter)
            model_executor.ModelExecutor(
                config, model, _CacheConsumer(), pool, None, draft_model_runner=model
            )
    drafter, target = stopped.value.args
    assert target is model.model
    assert drafter.received_config == algorithm_config
    assert drafter.configure_calls == (0 if algorithm_config is None else 1)
