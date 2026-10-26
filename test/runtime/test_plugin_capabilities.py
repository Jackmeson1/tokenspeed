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

"""Optional drafter contracts preserve native behavior when omitted."""

import importlib

import pytest
import torch

from tokenspeed.runtime.execution.drafter.base import (
    BaseDrafter,
    reserved_hook_problems,
)
from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ, TreeVerifyBatch


def test_default_round_hooks_leave_drafting_in_the_native_path():
    drafter = BaseDrafter(spec_num_tokens=4)
    drafter.prepare_round_sampling([])
    drafter.capture_round_graphs(4, lo_bs=2, stream=None, pool=(0, 0))
    drafter.finish_round(None, None)
    drafter.complete_round()
    drafter.release_round_graphs()
    assert not drafter.has_round_graphs(4)
    assert not drafter.suspend_drafting(2, ["first", "second"])
    assert drafter.verify_widths(4) == (4,)
    assert drafter.round_verify_width(2) == 4
    assert reserved_hook_problems(BaseDrafter) == []


def test_default_replay_runs_the_supplied_graph_once():
    class Graph:
        # CUDA graph substitution: CPU tests cannot capture GPU launches.
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    graph = Graph()
    BaseDrafter(spec_num_tokens=4).replay_decode_graph(graph, bs=2, graph_bs=4)
    assert graph.replays == 1


@pytest.mark.parametrize(
    "method,capability",
    [
        ("suspend_drafting", "drafter.suspend_drafting"),
        ("verify_widths", "drafter.verify_widths"),
        ("round_verify_width", "drafter.verify_widths"),
    ],
)
def test_reserved_policy_override_is_rejected(method, capability):
    plugin = type("UnsupportedPolicy", (BaseDrafter,), {method: lambda self, *args: 1})
    problems = reserved_hook_problems(plugin)
    assert len(problems) == 1
    assert capability in problems[0]


@pytest.mark.parametrize(
    "module,name,can_verify_q",
    [
        ("base", "SamplingBackend", False),
        ("greedy", "GreedySamplingBackend", False),
        ("triton", "TritonSamplingBackend", False),
        ("triton_full", "TritonFullSamplingBackend", False),
        ("flashinfer_full", "FlashInferFullSamplingBackend", False),
        ("flashinfer", "FlashInferSamplingBackend", False),
    ],
)
def test_recorded_proposal_negotiation_uses_the_selected_backend(
    module, name, can_verify_q
):
    # A plugin queries this public flag before allocating or publishing q.
    backend = getattr(
        importlib.import_module(f"tokenspeed.runtime.sampling.backends.{module}"), name
    )
    assert backend.verifies_draft_tree_q is can_verify_q
    if can_verify_q:
        assert backend.supports_tree_verify


def test_recorded_proposals_are_optional_and_start_deterministic():
    parents = torch.tensor([[-1, 0, 1]], dtype=torch.int32)
    depths = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    assert TreeVerifyBatch(parents, depths).draft_q is None
    q = DraftTreeQ.zeros(2, 4, 5, 3, "cpu")
    assert q.q.shape == (2, 4, 5)
    assert torch.count_nonzero(q.q) == 0
    assert torch.all(q.slot == -1)
    q.q[1, 2, :2] = torch.tensor([0.75, 0.25])
    q.q_ids[1, 2, :2] = torch.tensor([11, 19], dtype=q.q_ids.dtype)
    q.slot[1, 1] = 2
    q.pos[1, 1] = 0
    tree = TreeVerifyBatch(parents, depths, q)
    assert tree.draft_q.q[1, tree.draft_q.slot[1, 1], 0] == 0.75
