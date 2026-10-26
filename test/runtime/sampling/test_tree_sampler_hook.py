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

"""Instance-local tree verification leaves unbound backends unchanged."""

import pytest
import torch

from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.flashinfer import FlashInferSamplingBackend
from tokenspeed.runtime.sampling.backends.flashinfer_full import (
    FlashInferFullSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ, TreeVerifyBatch
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.mark.parametrize("tree", [False, True])
def test_sampling_config_preserves_explicit_draft_probability_flag(tree):
    args = ServerArgs(model="unused")
    args.enable_speculative_sampling = True
    args.speculative_algorithm = "EAGLE3"
    args.speculative_eagle_topk = 2 if tree else 1
    assert args.speculative_draft_tree is tree
    config = SamplingBackendConfig.from_server_args(
        args, max_bs=2, max_draft_tokens_per_req=3, device="cpu"
    )
    assert config.enable_speculative_sampling


def _backend(cls, monkeypatch):
    # CUDA stream creation has no CPU equivalent. All backend state is real.
    monkeypatch.setattr(
        "tokenspeed.runtime.sampling.backends.flashinfer.fused_topk_topp_prepare",
        lambda device: None,
    )
    return cls(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=2,
            max_draft_tokens_per_req=3,
            max_req_pool_size=2,
            vocab_size=5,
            device="cpu",
            enable_tp_sync=False,
        )
    )


class RootSampler:
    """A target-only verifier: emit the root argmax without accepting drafts."""

    def __init__(self, pools, max_bs, device):
        assert max_bs == 2
        assert str(device) == "cpu"
        assert pools.temperature.shape == (3,)

    def __call__(
        self,
        logits,
        candidates,
        tree,
        req_pool_indices,
        *,
        coins,
        final_coins,
        path,
        predict,
        accept_length
    ):
        bs, n = candidates.shape
        assert coins.shape == (bs, n)
        assert final_coins.shape == (bs,)
        path.fill_(-1)
        path[:, 0].zero_()
        predict.zero_()
        predict.view(bs, n)[:, 0].copy_(logits.view(bs, n, -1)[:, 0].argmax(-1))
        accept_length.fill_(1)


@pytest.mark.parametrize("recorded_q", [False, True])
def test_bound_sampler_verifies_without_changing_another_instance(
    monkeypatch, recorded_q
):
    native = _backend(FlashInferSamplingBackend, monkeypatch)
    plugin = _backend(FlashInferSamplingBackend, monkeypatch)
    BaseDrafter(spec_num_tokens=3).bind_sampling_backend(native)
    plugin.bind_tree_sampler(RootSampler)
    tree = TreeVerifyBatch(
        torch.tensor([[-1, 0, 1]], dtype=torch.int32),
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        DraftTreeQ.zeros(3, 2, 5, 3, "cpu") if recorded_q else None,
    )
    logits = LogitsProcessorOutput(
        next_token_logits=torch.tensor(
            [
                [0.0, 1.0, 4.0, 2.0, 3.0],
                [3.0, 1.0, 2.0, 4.0, 0.0],
                [1.0, 2.0, 3.0, 4.0, 0.0],
            ]
        )
    )
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([1]),
        vocab_size=5,
        device="cpu",
        batch_row_offset=1,
    )
    candidates = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match="cannot verify draft trees"):
        native.verify(logits, info, candidates, tree=tree)
    predictions, lengths = plugin.verify(logits, info, candidates, tree=tree)
    assert predictions.tolist() == [2, 0, 0]
    assert lengths.tolist() == [1]
    assert plugin.accepted_path(1, 3).tolist() == [[0, -1, -1]]
    assert plugin.supports_tree_verify and plugin.verifies_draft_tree_q
    assert not native.supports_tree_verify and not native.verifies_draft_tree_q
    with pytest.raises(RuntimeError, match="only once"):
        plugin.bind_tree_sampler(RootSampler)


@pytest.mark.parametrize("cls", [GreedySamplingBackend, FlashInferFullSamplingBackend])
def test_unsupported_backends_refuse_sampler_binding(monkeypatch, cls):
    backend = _backend(cls, monkeypatch)
    with pytest.raises(NotImplementedError, match="cannot bind"):
        backend.bind_tree_sampler(RootSampler)
