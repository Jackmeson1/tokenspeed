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

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import torch
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax

from tokenspeed.runtime.execution.drafter.speculative_sampling import (
    DraftProposalSampler,
)
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.slot_state import (
    pack_slot_rows,
    slot_state_image_bytes,
    unpack_slot_rows,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.execution.tree_spec import TreeSpec
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.base import SamplingBackend
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class BaseDrafter:
    #: Slot-sized tensors a retraction image leaves out (``SlotStateExporter``):
    #: none by default; a drafter that keeps some names them.
    token_derived_slot_state: tuple[str, ...] = ()
    constant_slot_state: tuple[str, ...] = ()

    # Whether the draft model reuses the target's embedding and LM head
    # weights (set via set_embed_and_head right after both models load, in
    # create_model_runner, so the draft's own copies are dropped before the
    # KV-cache budget is profiled).
    shares_target_embed_head = False

    # True only when ``run`` returns after every cache-writing draft pass has
    # been enqueued on the caller's CUDA stream. CachePD uses that guarantee to
    # publish one final readiness event for the complete speculative chain.
    supports_pd_layerwise_finalization = False

    # Whether this drafter threads a request-token-history view into every
    # draft forward. A draft model whose config requires the history must be
    # served by a drafter that declares this, or fail at startup instead of
    # forwarding without it.
    supports_request_token_history = False

    # Whether this drafter proposes one token per step from its own logits
    # and can therefore sample it from the recorded distribution q
    # (--enable-speculative-sampling). Block drafters propose a whole block
    # greedily and leave this False.
    supports_speculative_sampling = False

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int | None = None,
        draft_model_runner: ModelRunner | None = None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        attn_backend: AttentionBackend | None = None,
        token_to_kv_pool: CachePool | None = None,
        vocab_size: int | None = None,
    ):
        self.spec_num_tokens = spec_num_tokens
        self.spec_num_steps = spec_num_steps
        self.draft_model_runner = draft_model_runner
        self.runtime_states = runtime_states
        self.input_buffers = input_buffers
        # This round's per-group scheduler tables, published by the runner's
        # metadata prep; block drafters re-run the unified refresh with them
        # inside their step loop (write locations themselves come from the
        # draft router: publish_draft_step_locations serves the step window,
        # draft_write_locations_uniform resolves side-write scratch).
        self.round_block_tables = None
        self.attn_backend = attn_backend
        self.token_to_kv_pool = token_to_kv_pool
        self.vocab_size = vocab_size
        # Bound by ``bind_sampling_backend`` under --enable-speculative-sampling;
        # None keeps the argmax proposal.
        self.draft_sampler: DraftProposalSampler | None = None

    def set_cache_pool(self, token_to_kv_pool: CachePool | None) -> None:
        """Take a replacement pool; a drafter that caches views rebuilds them."""
        self.token_to_kv_pool = token_to_kv_pool

    def configure_algorithm(self, algorithm_config: Mapping[str, Any]) -> None:
        """Take the launch's ``--speculative-algorithm-config``.

        Called once by ``ModelExecutor`` right after construction, before
        ``wire_target``, ``bind_sampling_backend`` and ``bind_tree``, and only
        when the launch passed a config. Server arguments refuse the flag for
        the in-tree algorithms, so only a plugin drafter receives one. The
        default refuses it: a drafter that reads no settings must not drop
        them silently.

        Args:
            algorithm_config: The parsed JSON object, uninterpreted.

        Raises:
            ValueError: This drafter takes no algorithm config.
        """
        raise ValueError(
            f"{type(self).__name__} takes no --speculative-algorithm-config; "
            f"got keys {sorted(algorithm_config)}"
        )

    def bind_sampling_backend(self, sampling_backend: SamplingBackend) -> None:
        """Take the verifier's per-request sampling pools for sampled drafting.

        Called once by ``ModelExecutor`` after construction. A no-op unless
        the executor allocated ``RuntimeStates.draft_probs``
        (--enable-speculative-sampling); then the drafter must be a chain
        drafter that proposes one token per step, and its proposals are
        drawn from ``q`` and recorded there (``sample_draft_step``). Sampling
        needs the whole distribution, so the draft model's logits processor
        is switched off its fused (TP-sharded) argmax here, before the first
        draft forward.

        Args:
            sampling_backend: The executor's verifier, owner of the
                pool-indexed temperature / top-k / seed scalars.

        Raises:
            ValueError: The flag is on but this drafter proposes greedily.
        """
        if self.runtime_states is None or self.runtime_states.draft_probs is None:
            return
        if not self.supports_speculative_sampling:
            raise ValueError(
                "--enable-speculative-sampling needs a chain drafter that samples "
                f"one token per step from its own distribution; "
                f"{type(self).__name__} proposes a whole block greedily"
            )
        self.draft_sampler = DraftProposalSampler(
            pools=sampling_backend.speculative_sampling_pools(),
            runtime_states=self.runtime_states,
            input_buffers=self.input_buffers,
            spec_num_tokens=self.spec_num_tokens,
            vocab_map=self.draft_vocab_map(),
            device=self.runtime_states.draft_probs.device,
        )
        self.draft_model_runner.model.logits_processor.require_full_vocab_logits()

    def draft_vocab_map(self) -> torch.Tensor | None:
        """``[V_draft]`` int64 full-vocab id per draft logit column, or None
        when the draft head spans the target vocabulary. Eagle3 hot-token
        heads override this; ``sample_draft_step`` returns ids in the draft
        vocab either way and the caller maps them."""
        return None

    def sample_draft_step(
        self,
        logits_output: LogitsProcessorOutput,
        *,
        step: int,
    ) -> torch.Tensor:
        """One draft step's proposed token ids, ``[bs]`` in the draft vocab.

        Without draft-prob sampling this is the logits processor's fused
        argmax when it ran, else the canonical argmax of the logits. With
        it, the token is sampled from ``q`` at the request's temperature
        (greedy rows keep the argmax) and ``q`` is recorded for verify. The
        row count is the logits' (the padded graph batch under replay).

        Args:
            logits_output: The step's draft forward output, one row per
                request.
            step: Draft step index (verify candidate column ``step + 1``).
        """
        sampler = self.draft_sampler
        if sampler is None:
            if (
                self.runtime_states is not None
                and self.runtime_states.draft_probs is not None
            ):
                raise RuntimeError(
                    f"{type(self).__name__} records no draft distributions: "
                    "bind_sampling_backend was not called"
                )
            if logits_output.next_token_ids is not None:
                return logits_output.next_token_ids
            return sampling_argmax(logits_output.next_token_logits)
        return sampler.propose(logits_output.next_token_logits, step=step)

    def wire_target(self, target_model: torch.nn.Module) -> None:
        """Wire this drafter to the loaded target model.

        Called once by ``ModelExecutor`` right after the drafter is
        constructed. Subclasses bind execution resources such as target weights
        and output heads here. Capture configuration belongs to model setup
        before drafter construction; this method must not change it. The
        default drafter needs nothing from the target.

        Args:
            target_model: The target ``torch.nn.Module`` the drafter
                speculates for.
        """

    def prepare_target_forward(self, ctx: ForwardContext) -> None:
        """Hook before the round's target forward.

        Called by ``ModelExecutor`` with the context the target is about to
        run under. A drafter that wants work done while the target runs
        attaches it to ``ctx`` here (DFLASH attaches its incremental capture
        sink, ``ctx.target_capture_sink``); the context is per forward, so
        nothing attached outlives the round. The default needs nothing.

        Args:
            ctx: The target forward's context.
        """

    def on_target_weights_updated(self) -> None:
        """Refresh draft-side state derived from shared target weights.

        Called after an in-place target weight update completes and before the
        device thread accepts another forward. Most drafters do not cache
        derived target weights and therefore need no action.
        """

    def idle_forward_global_num_tokens(
        self,
        global_num_tokens: list[int],
        global_bs: list[int],
        *,
        global_num_decodes: list[int],
    ) -> list[list[int]]:
        """The IDLE draft forwards an idle attention-DP rank runs per round,
        as the per-rank token counts each one reports: entry ``i`` sizes the
        forward with ``spec_step_idx=i``, and the list length is the number
        of draft forwards the active ranks run.

        The idle rank runs no rows of its own; it mirrors the collective
        sizing of the ranks that do, so the list follows the drafter's step
        loop. The default is the Eagle chain: step 0 runs the target's rows
        (its verify window per decode request), steps 1+ one row per request
        (``global_bs``). Drafters with another loop override this — block
        drafters run one forward, multi-depth MTP the target's rows at every
        depth.

        Args:
            global_num_tokens: The round's per-rank target token counts.
            global_bs: The round's per-rank total request counts.
            global_num_decodes: The round's per-rank decode request counts.

        Returns:
            One ``global_num_tokens`` per IDLE draft forward, in step order.
        """
        steps = int(self.spec_num_steps or 0)
        if steps == 0:
            return []
        return [global_num_tokens] + [global_bs] * (steps - 1)

    @property
    def captures_prefill_graph(self) -> bool:
        """Whether ``capture_prefill_graph`` records one, for the projection."""
        return False

    def release_prefill_graph(self) -> None:
        """Drop a captured draft prefill graph. Drafters without one need no action."""

    def capture_prefill_graph(
        self, stream: torch.cuda.Stream, observer: AbstractContextManager[None]
    ) -> None:
        """Capture draft prefill work after target capture, when prefill graphs
        are enabled. Drafters without a separate prefill graph need no action.
        """

    def bind_tree(self, tree_spec: TreeSpec) -> None:
        """Draft trees (--speculative-eagle-topk > 1); only EAGLE-style drafters expand lanes."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot draft trees (--speculative-eagle-topk > 1); "
            "tree drafting needs an EAGLE-style drafter (EAGLE3, or MTP served by Eagle)"
        )

    # ------------------------------------------------------------------
    # Slot-state image (SlotStateExporter)
    # ------------------------------------------------------------------

    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        """The drafter-owned per-slot rows a retraction snapshot images.

        Cross-round state keyed by ``req_pool_index`` that a forward alone
        regenerates -- a stash of target hiddens, a request-persistent
        context window, a write frontier -- belongs here, in a fixed order.
        A drafter whose state lives entirely in cache groups or batch rows
        keeps the empty default. Draft KV pages are cache groups and travel
        with the pages, not here.
        """
        del slot
        return []

    def slot_state_bytes(self) -> int:
        return slot_state_image_bytes(self.slot_state_rows(0))

    def export_slot_state(
        self, slot: int, out: torch.Tensor, stream, *, request_id: str
    ) -> None:
        # The default rows are rewritten by every forward of the slot, never
        # keyed by request; a drafter that keys them (DSpark) overrides both
        # copies to image the prepared marker (``slot_state.py``).
        del request_id
        pack_slot_rows(self.slot_state_rows(slot), out, stream)

    def import_slot_state(
        self, slot: int, src: torch.Tensor, stream, *, request_id: str
    ) -> None:
        del request_id
        unpack_slot_rows(self.slot_state_rows(slot), src, stream)

    @abstractmethod
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def rewrite_verify_outputs(
        self,
        ctx: ForwardContext,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> None:
        """Rewrite the verify's outputs in place before anything reads them."""

    def prepare_round_sampling(
        self, sampling_params_list: list[SamplingParams]
    ) -> None:
        """Take the forward's sampling parameters before it runs (host-side).
        Default: ignored."""

    def replay_decode_graph(
        self, graph: torch.cuda.CUDAGraph, *, bs: int, graph_bs: int
    ) -> None:
        """Replay the target's decode graph for ``bs`` live requests padded to
        ``graph_bs``, the forward's stream ordered after it; a drafter may
        start its round behind it on another stream. Default: replays it on
        the current stream."""
        del bs, graph_bs
        graph.replay()

    def finish_round(self, ctx: ForwardContext, graph_bs: int | None) -> None:
        """Draft after the forward at its decode graph's padded batch (None: eager)."""

    def complete_round(self) -> None:
        """Finish a round ``finish_round`` left pending (the next round's inputs)."""

    def capture_round_graphs(
        self,
        graph_bs: int,
        *,
        lo_bs: int,
        stream: torch.cuda.Stream,
        pool: tuple[int, int],
    ) -> None:
        """Capture graphs serving ``lo_bs < live_bs <= graph_bs`` in this pool."""

    def has_round_graphs(self, graph_bs: int) -> bool:
        """Whether a completed prefill can use round graphs at this padded batch."""
        return False

    def release_round_graphs(self) -> None:
        """Drop the graphs ``capture_round_graphs`` recorded (they name the cache pool)."""

    def suspend_drafting(self, num_reqs: int, req_ids: Sequence[str]) -> bool:
        """Whether to decode this batch without drafting.

        Called on the host once per scheduled decode batch, before its inputs
        are built, with the batch's request count and ids; it must not read
        device memory. True suspends drafting for every request of the batch
        for the rest of its life: they decode one token per round through
        target-only graphs and never draft again. The default never suspends.
        Capability: ``drafter.suspend_drafting``.
        """
        del num_reqs, req_ids
        return False

    def verify_widths(self, graph_bs: int) -> tuple[int, ...]:
        """The verify widths to capture target decode graphs for at padded
        batch ``graph_bs``, each in ``[2, spec_num_tokens]``.

        Called once per captured batch size. A round verifying width ``w``
        verifies the root and the first ``w - 1`` drafted tokens of each
        request's window. The default is the configured width only.
        Capability: ``drafter.verify_widths``.
        """
        del graph_bs
        return (self.spec_num_tokens,)

    def round_verify_width(self, bs: int) -> int:
        """The verify width of the next decode round for ``bs`` live requests:
        one of ``verify_widths`` of its padded batch.

        Called on the host before each decode replay, only when
        ``verify_widths`` is overridden; it must not read device memory. The
        default is the configured width. Capability: ``drafter.verify_widths``.
        """
        del bs
        return self.spec_num_tokens

    @abstractmethod
    def draft(self, *args, **kwargs) -> torch.Tensor | None:
        raise NotImplementedError


# Hooks whose mechanism this runtime does not implement yet: a drafter that
# overrides one is refused at startup instead of being silently ignored.
_RESERVED_HOOKS = {
    "suspend_drafting": "drafter.suspend_drafting",
    "verify_widths": "drafter.verify_widths",
    "round_verify_width": "drafter.verify_widths",
}


def reserved_hook_problems(drafter_cls: type) -> list[str]:
    """The reserved ``BaseDrafter`` hooks ``drafter_cls`` overrides although
    this runtime does not advertise their capability."""
    from tokenspeed.runtime.plugins import PLUGIN_CAPABILITIES

    return [
        f"{drafter_cls.__name__}.{name} needs plugin capability {capability!r}, "
        "which this runtime does not implement"
        for name, capability in _RESERVED_HOOKS.items()
        if capability not in PLUGIN_CAPABILITIES
        and getattr(drafter_cls, name) is not getattr(BaseDrafter, name)
    ]
