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

"""Round-local lane tables must null padding without changing step-0 buffers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.paged import trtllm
from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeDraftInputs,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.paged_attention import PagedAttention

register_cuda_ci(est_time=5, suite="runtime-1gpu")

MAX_BS, K, N = 4, 2, 4
PAGES = {"full": 64, "other": 128}


def _router(*, device: str, bind_before_init: bool):
    spec = MHAConfig(
        backend_name="trtllm",
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=128,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=128,
        context_len=256,
        max_bs=MAX_BS,
        speculative_num_steps=3,
        speculative_num_draft_tokens=N,
        is_draft=True,
        components=(spec,),
    )
    leaves = {
        gid: trtllm.TRTLLMMHAAttnBackend(config, spec, kernel_page_size=page)
        for gid, page in PAGES.items()
    }
    router = CacheGroupRouter(
        None,
        is_draft=True,
        spec_num_tokens=N,
        device=device,
        consumed_group_ids=None,
    )
    router.bind(
        CacheGroupGeometry(
            granularities=PAGES,
            attn_kinds={gid: "full_history" for gid in PAGES},
            full_history_group_id="full",
            row_geometry={gid: (page, 1) for gid, page in PAGES.items()},
            windows={gid: None for gid in PAGES},
        ),
        leaves,
    )
    lanes = TreeDraftInputs(K, 3, MAX_BS, torch.device(device))
    if bind_before_init:
        router.bind_tree_draft(lanes)
    router.init_cuda_graph_state(MAX_BS)
    if not bind_before_init:
        router.bind_tree_draft(lanes)
    lanes.set_frontier(
        MAX_BS, torch.full((MAX_BS,), 3, dtype=torch.int32, device=device)
    )
    return router, lanes


def _layer(gid: str) -> PagedAttention:
    layer = PagedAttention(
        num_heads=2,
        head_dim=128,
        scaling=128**-0.5,
        num_kv_heads=1,
        layer_id=0,
        sliding_window_size=16,
        rotary_emb=None,
        qk_norm=None,
    )
    layer.bind_cache_group(gid)
    return layer


def _refresh(router, bs: int, page: int):
    # Preparing only live rows reproduces pure extend followed by a padded graph.
    tables = {
        gid: torch.full(
            (bs, leaf.max_num_pages), page + i, dtype=torch.int32, device=router.device
        )
        for i, (gid, leaf) in enumerate(router.leaves.items())
    }
    router.refresh_decode_metadata(
        bs,
        bs,
        torch.arange(bs, device=router.device),
        torch.full((bs,), 8, dtype=torch.int32, device=router.device),
        forward_mode=ForwardMode.DECODE,
        block_tables=tables,
    )


def _publish(router, lanes, bs: int):
    lanes.active = True
    return router.publish_draft_step_locations(lanes.frontier[:bs], K)


@pytest.mark.parametrize("bs,live_bs", [(4, 3), (4, 1), (2, 1), (4, 0)])
def test_shrinking_round_nulls_reads_and_writes_without_changing_old_buffers(
    bs, live_bs
):
    router, lanes = _router(device="cpu", bind_before_init=False)
    _refresh(router, MAX_BS, 7)
    router.begin_tree_lanes(MAX_BS, MAX_BS)
    _publish(router, lanes, MAX_BS)
    lanes.active = False
    _refresh(router, live_bs, 2)
    original = [router.stacks.tables, router.stacks.decode_locs]
    original += [leaf.page_table_buf for leaf in router.leaves.values()]
    saved = [buf.clone() for buf in original]
    published = router.decode_write_locations

    router.begin_tree_lanes(bs, live_bs)
    _publish(router, lanes, bs)
    for i, (gid, leaf) in enumerate(router.leaves.items()):
        rows = leaf._tree_lane_rows(bs)
        assert torch.all(rows.page_table[:live_bs, : leaf.max_num_pages] == 2 + i)
        assert torch.count_nonzero(rows.page_table[live_bs:]) == 0
        locs = router.forward_write_locations(_layer(gid), ForwardMode.DECODE)
        expected = torch.tensor([(2 + i) * PAGES[gid] + 3, (2 + i) * PAGES[gid] + 4])
        torch.testing.assert_close(
            locs[: live_bs * K].view(live_bs, K),
            expected.expand(live_bs, K).to(torch.int32),
        )
        assert torch.count_nonzero(locs[live_bs * K :]) == 0
    assert router.decode_write_locations is published
    for buf, before in zip(original, saved):
        torch.testing.assert_close(buf, before)


def test_recorded_lane_views_survive_decode_refresh_and_next_round():
    router, lanes = _router(device="cpu", bind_before_init=True)
    _refresh(router, MAX_BS, 2)
    router.begin_tree_lanes(MAX_BS, MAX_BS)
    locations = _publish(router, lanes, MAX_BS)
    tables = router.leaves["full"]._tree_lane_rows(MAX_BS).page_table
    saved_tables, saved_locations = tables.clone(), locations.clone()
    pointers = tables.data_ptr(), locations.data_ptr()
    lanes.active = False
    _refresh(router, 1, 9)
    torch.testing.assert_close(tables, saved_tables)
    torch.testing.assert_close(locations, saved_locations)

    router.begin_tree_lanes(MAX_BS, 1)
    next_locations = _publish(router, lanes, MAX_BS)
    assert (tables.data_ptr(), next_locations.data_ptr()) == pointers
    assert locations.tolist() == [579, 580, 0, 0, 0, 0, 0, 0]
    assert torch.all(tables[0] == 9)
    assert torch.count_nonzero(tables[1:]) == 0


def test_pure_extend_followed_by_padded_lanes_uses_live_router_tables():
    router, lanes = _router(device="cpu", bind_before_init=False)
    _refresh(router, 4, 7)
    lengths = torch.full((3,), 8, dtype=torch.int32)
    prefixes = torch.zeros_like(lengths)
    tables = {
        gid: torch.full((3, leaf.max_num_pages), 2, dtype=torch.int32)
        for gid, leaf in router.leaves.items()
    }
    router.init_forward_metadata(
        3,
        3,
        torch.arange(3),
        lengths,
        ForwardMode.EXTEND,
        extend_seq_lens=lengths,
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens=prefixes,
        extend_prefix_lens_cpu=prefixes,
        extend_replay_lens_cpu=prefixes,
        extend_prompt_lens_cpu=lengths,
        extend_with_prefix=False,
        query_shard=None,
        block_tables=tables,
        block_tables_cpu=tables,
    )
    router.begin_tree_lanes(4, 3)
    locations = _publish(router, lanes, 4)
    assert locations.tolist() == [131, 132] * 3 + [0, 0]
    rows = router.leaves["full"]._tree_lane_rows(4)
    assert torch.all(rows.page_table[:3] == 2)
    assert torch.count_nonzero(rows.page_table[3]) == 0


@pytest.mark.parametrize("bind_before_init", [False, True])
def test_native_tree_without_snapshot_keeps_refresh_and_write_behavior(
    bind_before_init,
):
    router, lanes = _router(device="cpu", bind_before_init=bind_before_init)
    for page in (2, 9):
        lanes.active = False
        _refresh(router, 2, page)
        locations = _publish(router, lanes, 2)
        assert locations.tolist() == [page * 64 + 3, page * 64 + 4] * 2
        rows = router.leaves["full"]._tree_lane_rows(2)
        assert torch.all(rows.page_table == page)
        assert (
            rows.page_table.data_ptr()
            == router.leaves["full"].page_table_buf.data_ptr()
        )
        torch.testing.assert_close(
            router.write_locations(_layer("full"), ForwardMode.DECODE), locations
        )


@dataclass
class _ArenaMetadata:
    cache_group_specs: tuple[CacheGroupSpec, ...]
    offload: None = None


class _PoolMetadata:
    """Pool binding fixture without allocating model KV weights."""

    def __init__(self):
        self.paged_group_ids = tuple(PAGES)
        self.arena = _ArenaMetadata(
            tuple(
                CacheGroupSpec(
                    group_id=gid,
                    attn_kind="full_history",
                    rows_per_page=page,
                    entry_stride_tokens=1,
                    replayable=False,
                )
                for gid, page in PAGES.items()
            )
        )


@pytest.mark.parametrize("rebind", [False, True])
def test_graph_state_recreation_rebinds_snapshot_and_keeps_old_buffers(rebind):
    router, lanes = _router(device="cpu", bind_before_init=True)
    _refresh(router, 4, 7)
    router.begin_tree_lanes(4, 4)
    old_locations = _publish(router, lanes, 4)
    old_table = router.leaves["full"]._tree_lane_rows(4).page_table
    before_table, before_locations = old_table.clone(), old_locations.clone()
    lanes.active = False
    if rebind:
        router.set_cache_pool(_PoolMetadata())
    router.init_cuda_graph_state(MAX_BS)
    _refresh(router, 1, 2)
    # Recreated state has no active snapshot until the new round starts it.
    assert _publish(router, lanes, 1).tolist() == [131, 132]
    assert torch.all(router.leaves["full"]._tree_lane_rows(1).page_table == 2)
    lanes.active = False
    router.begin_tree_lanes(4, 1)
    locations = _publish(router, lanes, 4)
    assert locations.tolist() == [131, 132, 0, 0, 0, 0, 0, 0]
    assert (
        router.leaves["full"]._tree_lane_rows(4).page_table.data_ptr()
        != old_table.data_ptr()
    )
    torch.testing.assert_close(old_table, before_table)
    torch.testing.assert_close(old_locations, before_locations)


@pytest.mark.parametrize("bs,live_bs", [(4, 5), (5, 4), (4, -1)])
def test_invalid_round_bounds_leave_snapshot_unchanged(bs, live_bs):
    router, lanes = _router(device="cpu", bind_before_init=True)
    _refresh(router, 4, 7)
    router.begin_tree_lanes(4, 4)
    _publish(router, lanes, 4)
    table = router.leaves["full"]._tree_lane_rows(4).page_table
    before = table.clone()
    with pytest.raises(ValueError):
        router.begin_tree_lanes(bs, live_bs)
    torch.testing.assert_close(table, before)


class _Composite(AttentionBackend):
    def __init__(self, router):
        self.router = router

    def child_backends(self):
        return (self.router,)


def test_composite_forwards_snapshot_to_router():
    router, lanes = _router(device="cpu", bind_before_init=False)
    _refresh(router, 4, 7)
    _refresh(router, 1, 2)
    _Composite(router).begin_tree_lanes(4, 1)
    assert _publish(router, lanes, 4).tolist() == [131, 132, 0, 0, 0, 0, 0, 0]


class _KVRows:
    """Small cache fixture; no model weights or GPU kernel are needed."""

    def __init__(self):
        self.k = torch.zeros((128, 1, 128), dtype=torch.bfloat16)
        self.v = torch.zeros_like(self.k)

    def get_kv_buffer(self, layer_id):
        return self.k, self.v


@pytest.mark.parametrize("snapshot", [False, True])
def test_sliding_window_lane_tables_keep_the_prefix_window(monkeypatch, snapshot):
    router, lanes = _router(device="cpu", bind_before_init=False)
    _refresh(router, 4, 7)
    _refresh(router, 1, 2)
    if snapshot:
        router.begin_tree_lanes(4, 1)
    _publish(router, lanes, 4)
    leaf = router.leaves["full"]
    seen = []

    def prefix_decode(
        *, query, block_tables, seq_lens, window_left, q_len_per_req, **kwargs
    ):
        # Stand in only for the unavailable GPU kernel; expose the table it reads.
        seen.append((block_tables, seq_lens.clone(), window_left, q_len_per_req))
        values = block_tables[:, 0].to(query.dtype)
        return values[:, None, None].expand_as(query), torch.zeros(query.shape[:2])

    def window_decode(q, k, v, table, seq_lens, mask, prefix_out, prefix_lse, **kwargs):
        assert torch.equal(table.repeat_interleave(K, dim=0), seen[-1][0])
        assert seq_lens.tolist() == [7] * 4
        assert kwargs["causal_prefix"] is False
        return prefix_out

    monkeypatch.setattr(trtllm, "trtllm_batch_decode_with_kv_cache", prefix_decode)
    monkeypatch.setattr(trtllm, "tree_window_attention", window_decode)
    query = torch.zeros((8, 2, 128), dtype=torch.bfloat16)
    pool, layer = _KVRows(), _layer("full")
    if not snapshot:
        with pytest.raises(NotImplementedError, match="sliding-window"):
            leaf.forward_decode(query, None, None, layer, None, pool, 4)
        return
    for step in (1, 2):
        lanes.step = step
        out = leaf.forward_decode(query, None, None, layer, None, pool, 4)
        assert out[:, 0].tolist() == [2, 2] + ([0] * 6 if snapshot else [7] * 6)
        assert seen[-1][1].tolist() == [3] * 8
        assert seen[-1][2:] == (16, 1)
    if snapshot:
        assert seen[0][0].data_ptr() == seen[1][0].data_ptr()
    assert torch.all(leaf.page_table_buf[1:] == 7)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph replay requires a GPU"
)
@pytest.mark.parametrize("detached", [False, True])
def test_cuda_graph_replay_uses_current_round_snapshot(detached):
    router, lanes = _router(device="cuda", bind_before_init=True)
    _refresh(router, 4, 7)
    router.begin_tree_lanes(4, 4)
    _publish(router, lanes, 4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if not detached:
            router.begin_tree_lanes(4, 1)
        locations = _publish(router, lanes, 4).clone()
        table = router.leaves["full"]._tree_lane_rows(4).page_table.clone()
    for page in (2, 9):
        lanes.active = False
        _refresh(router, 1, page)
        if detached:
            router.begin_tree_lanes(4, 1)
        graph.replay()
        assert locations.cpu().tolist() == [
            page * 64 + 3,
            page * 64 + 4,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        assert torch.all(table[0] == page)
        assert torch.count_nonzero(table[1:]) == 0
