# The previous Eagle 4 code merged to TokenSpeed was in violation of the Eagle 4 source
# available license. This PR overwrites these previous violating PRs for Eagle 4 in
# TokenSpeed. TokenSpeed and Eagle agree that this PR is the only permitted Eagle 4
# branch for the TokenSpeed codebase, and any previous Eagle 4 code in TokenSpeed is
# effectively cancelled.
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

"""Drafter registry: resolve a speculative algorithm to its drafter class.

The in-tree algorithms stay lazily imported (drafter modules pull in kernel
ops and model code, and this package init must stay importable from
lightweight contexts). Plugins extend the table through
:func:`register_drafter`: either a new algorithm name, or a draft-model-class
scoped entry that specializes an existing algorithm the way the in-tree
Inkling/DFlash2/DSpark cases do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tokenspeed.runtime.execution.drafter.base import BaseDrafter
    from tokenspeed.runtime.utils.server_args import ServerArgs

__all__ = [
    "IN_TREE_ALGORITHMS",
    "drafter_captures_aux_hidden_states",
    "drafter_uses_tree_layout",
    "get_drafter_impl",
    "register_drafter",
    "registered_drafter_algorithms",
    "require_plugin_draft_checkpoint",
    "validate_drafter_algorithm",
]

#: Algorithms the in-tree resolution below serves.
IN_TREE_ALGORITHMS = frozenset({"EAGLE3", "MTP", "DFLASH", "DSPARK", "DUMMY"})


@dataclass
class _AlgorithmEntries:
    """Plugin entries for one algorithm: class-scoped first, then a default."""

    scoped: list[tuple[type, type]] = field(default_factory=list)
    default: type | None = None
    defaults_to_base_checkpoint: bool = False
    captures_aux_hidden_states: bool = False
    uses_tree_layout: bool = False


_PLUGIN_DRAFTERS: dict[str, _AlgorithmEntries] = {}


def register_drafter(
    algorithm: str,
    drafter_cls: type,
    *,
    model_cls: type | None = None,
    defaults_to_base_checkpoint: bool = False,
    captures_aux_hidden_states: bool = False,
    uses_tree_layout: bool = False,
    override: bool = False,
) -> None:
    """Register a drafter class for a speculative algorithm.

    Args:
        algorithm: The ``--speculative-algorithm`` name. A new name adds an
            algorithm; an in-tree name may only be extended with a
            ``model_cls``-scoped entry (or replaced with ``override=True``).
        drafter_cls: The ``BaseDrafter`` subclass to instantiate.
        model_cls: When given, the entry applies only to draft models that
            are instances of this class; scoped entries win over the
            algorithm default, most recently registered first.
        defaults_to_base_checkpoint: Whether this algorithm's draft weights
            live in the base checkpoint. Server-args resolution runs before
            plugins load, so launches still pass ``--draft-model-path-use-base``;
            one that omits every draft path is refused with that instruction
            (``require_plugin_draft_checkpoint``).
        captures_aux_hidden_states: Whether this algorithm's draft reads the
            target's auxiliary hidden states, as EAGLE3 does: the target then
            captures the layers ``--eagle3-layers-to-capture`` or the draft
            config's ``eagle_aux_hidden_state_layer_ids`` name, and the draft
            receives them in ``logits_output.hidden_states``. This is an
            algorithm-wide contract declared by its default registration.
            Model-scoped entries inherit it and cannot enable capture for an
            algorithm whose default does not capture.
        uses_tree_layout: Whether this algorithm's drafter drafts with
            explicit parents and draft lanes even at ``--speculative-eagle-topk
            1``: the engine then binds a tree (``BaseDrafter.bind_tree``) for
            chains too. Algorithm-wide like ``captures_aux_hidden_states``;
            read after plugin discovery, before cache sizing and model
            construction.
        override: Allow replacing an existing default entry (in-tree or
            plugin). Overrides are logged by the plugin loader.

    Raises:
        ValueError: An unscoped registration collides without ``override``.
    """
    from tokenspeed.runtime.utils.server_args import AUX_CAPTURE_SPEC_ALGORITHMS

    existing = _PLUGIN_DRAFTERS.get(algorithm)
    captures_aux = drafter_captures_aux_hidden_states(algorithm)
    new_captures_aux = (
        algorithm in AUX_CAPTURE_SPEC_ALGORITHMS or captures_aux_hidden_states
    )
    if model_cls is not None and captures_aux_hidden_states and not captures_aux:
        raise ValueError(
            "auxiliary capture must be declared on the algorithm default "
            "before registering model-scoped drafters"
        )
    tree_layout = drafter_uses_tree_layout(algorithm)
    if model_cls is not None and uses_tree_layout and not tree_layout:
        raise ValueError(
            "the tree layout must be declared on the algorithm default "
            "before registering model-scoped drafters"
        )
    if model_cls is None and existing is not None and existing.scoped:
        if new_captures_aux != captures_aux:
            raise ValueError(
                "cannot change auxiliary capture while model-scoped drafters "
                "inherit the algorithm contract"
            )
        if uses_tree_layout != tree_layout:
            raise ValueError(
                "cannot change the tree layout while model-scoped drafters "
                "inherit the algorithm contract"
            )
    if model_cls is None:
        taken = (existing is not None and existing.default is not None) or (
            algorithm in IN_TREE_ALGORITHMS
        )
        if taken and not override:
            raise ValueError(
                f"drafter algorithm {algorithm!r} is already registered; pass "
                "override=True to replace it, or scope the entry with "
                "model_cls"
            )
    entries = _PLUGIN_DRAFTERS.get(algorithm)
    created = entries is None
    if entries is None:
        entries = _PLUGIN_DRAFTERS[algorithm] = _AlgorithmEntries()
    # Snapshot for the rollback: undoing must RESTORE what an override
    # replaced (an earlier plugin's default, the base-checkpoint flag), not
    # blank it — undos run in reverse registration order, so each one puts
    # back exactly the state it saw.
    prior_default = entries.default
    prior_defaults_to_base = entries.defaults_to_base_checkpoint
    prior_captures_aux = entries.captures_aux_hidden_states
    prior_tree_layout = entries.uses_tree_layout
    if model_cls is not None:
        entries.scoped.insert(0, (model_cls, drafter_cls))
    else:
        entries.default = drafter_cls
    if defaults_to_base_checkpoint:
        entries.defaults_to_base_checkpoint = True
    if model_cls is None:
        entries.captures_aux_hidden_states = new_captures_aux
        entries.uses_tree_layout = uses_tree_layout

    def undo() -> None:
        if model_cls is not None:
            try:
                entries.scoped.remove((model_cls, drafter_cls))
            except ValueError:
                pass
        else:
            entries.default = prior_default
        entries.defaults_to_base_checkpoint = prior_defaults_to_base
        entries.captures_aux_hidden_states = prior_captures_aux
        entries.uses_tree_layout = prior_tree_layout
        if created and entries.default is None and not entries.scoped:
            _PLUGIN_DRAFTERS.pop(algorithm, None)

    from tokenspeed.runtime.plugins import registry as plugin_registry

    plugin_registry.record_external(
        "drafter",
        algorithm if model_cls is None else f"{algorithm}[{model_cls.__name__}]",
        undo,
    )


def registered_drafter_algorithms() -> frozenset[str]:
    """Every speculative algorithm name, in-tree and plugin-registered."""
    return IN_TREE_ALGORITHMS | frozenset(_PLUGIN_DRAFTERS)


def drafter_captures_aux_hidden_states(algorithm: str | None) -> bool:
    """Whether ``algorithm``'s draft reads the target's auxiliary hidden states.

    True for EAGLE3 and for a plugin algorithm registered with
    ``captures_aux_hidden_states=True``. Plugins must be loaded first.
    """
    # server_args imports this package, so the constant is imported lazily.
    from tokenspeed.runtime.utils.server_args import AUX_CAPTURE_SPEC_ALGORITHMS

    if algorithm is None:
        return False
    if algorithm in AUX_CAPTURE_SPEC_ALGORITHMS:
        return True
    entries = _PLUGIN_DRAFTERS.get(algorithm)
    return entries is not None and entries.captures_aux_hidden_states


def drafter_uses_tree_layout(algorithm: str | None) -> bool:
    """Whether ``algorithm``'s drafter needs explicit ancestry even for chains.

    True for a plugin algorithm registered with ``uses_tree_layout=True``.
    Plugins must be loaded first.
    """
    if algorithm is None:
        return False
    entries = _PLUGIN_DRAFTERS.get(algorithm)
    return entries is not None and entries.uses_tree_layout


def validate_drafter_algorithm(name: str | None) -> None:
    """Reject an unknown ``--speculative-algorithm`` after plugin discovery.

    Args:
        name: The launch's algorithm name, or None when speculation is off.

    Raises:
        ValueError: The name matches no registered algorithm.
    """
    if name is None:
        return
    known = registered_drafter_algorithms()
    if name not in known:
        raise ValueError(
            f"unknown --speculative-algorithm {name!r}; available: " f"{sorted(known)}"
        )


def require_plugin_draft_checkpoint(server_args: ServerArgs) -> None:
    """Refuse a plugin speculative launch that names no draft checkpoint.

    Server-args resolution defaults the draft path for the in-tree algorithms
    that read the base checkpoint, but it runs before plugins load, so a
    plugin algorithm registered with ``defaults_to_base_checkpoint`` cannot
    be defaulted there. Without a path the launch would build no draft model
    and fail deep in the executor; say what to pass instead.

    Args:
        server_args: The launch's resolved server arguments.

    Raises:
        ValueError: A plugin algorithm launched without a draft checkpoint.
    """
    name = server_args.speculative_algorithm
    entries = _PLUGIN_DRAFTERS.get(name) if name is not None else None
    if entries is None or name in IN_TREE_ALGORITHMS:
        return
    if server_args.speculative_draft_model_path is not None:
        return
    if entries.defaults_to_base_checkpoint:
        raise ValueError(
            f"--speculative-algorithm {name} reads its draft from the base "
            "checkpoint; pass --draft-model-path-use-base (server arguments "
            "resolve before plugins load, so the registration cannot default it)"
        )
    raise ValueError(
        f"--speculative-algorithm {name} needs a draft checkpoint: pass "
        "--speculative-draft-model-path or --draft-model-path-use-base"
    )


def get_drafter_impl(
    spec_algo: str, model: torch.nn.Module | None
) -> type[BaseDrafter]:
    """Resolve the drafter class for ``spec_algo`` and a loaded draft model.

    Args:
        spec_algo: The speculative algorithm name from server args.
        model: The loaded draft model; None for model-free DUMMY drafting.

    Returns:
        The ``BaseDrafter`` subclass to instantiate (not an instance).
    """
    plugin = _PLUGIN_DRAFTERS.get(spec_algo)
    if plugin is not None:
        for model_cls, drafter_cls in plugin.scoped:
            if isinstance(model, model_cls):
                return drafter_cls
        if plugin.default is not None:
            return plugin.default

    validate_drafter_algorithm(spec_algo)

    # Imports are local: drafter modules pull in kernel ops and model code,
    # and this package init must stay importable from lightweight contexts.
    from tokenspeed.runtime.execution.drafter.dflash import DFlash
    from tokenspeed.runtime.execution.drafter.dspark import DSpark
    from tokenspeed.runtime.execution.drafter.dummy import DummyDrafter
    from tokenspeed.runtime.execution.drafter.eagle import Eagle
    from tokenspeed.runtime.models.inkling_nextn import (
        InklingForConditionalGenerationNextN,
    )

    DRAFTER_MAPPING = {
        "EAGLE3": Eagle,
        "MTP": Eagle,
        "DFLASH": DFlash,
        "DSPARK": DSpark,
        "DUMMY": DummyDrafter,
    }

    if spec_algo == "DFLASH":
        from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2
        from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel

        if isinstance(model, DFlash2DraftModel):
            return DFlash2

    # "MTP" covers two algorithms:
    # (1) Eagle-like MTP (e.g. DeepSeek) stays on Eagle in eagle.py;
    # (2) Vanilla MTP (e.g. Inkling) with multi-layer weights stays on Mtp in mtp.py.
    if spec_algo == "DSPARK":
        from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import (
            DeepseekV41DSpark,
        )
        from tokenspeed.runtime.models.deepseek_v41_dspark import (
            DeepseekV41ForCausalLMDSpark,
        )

        if isinstance(model, DeepseekV41ForCausalLMDSpark):
            return DeepseekV41DSpark
    if spec_algo == "MTP" and isinstance(model, InklingForConditionalGenerationNextN):
        from tokenspeed.runtime.execution.drafter.mtp import Mtp

        return Mtp
    return DRAFTER_MAPPING[spec_algo]
