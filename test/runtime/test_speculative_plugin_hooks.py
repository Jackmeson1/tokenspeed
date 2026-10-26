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

"""Server arguments and drafter hooks of plugin-registered speculative algorithms."""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import argparse
import json
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution import drafter as drafter_registry
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.factory import configure_draft_target
from tokenspeed.runtime.plugins import registry
from tokenspeed.runtime.utils.server_args import ServerArgs

PLUGIN_ALGORITHM = "FIXTURE_SPEC"


def _args(**overrides) -> ServerArgs:
    kwargs = dict(
        model="x",
        speculative_algorithm=PLUGIN_ALGORITHM,
        speculative_draft_model_path="draft",
        sampling_backend="greedy",
    )
    kwargs.update(overrides)
    return ServerArgs(**kwargs)


def _cli(argv: list[str]) -> ServerArgs:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return ServerArgs.from_cli_args(parser.parse_args(["--model", "x", *argv]))


class _ConfiguredDrafter(BaseDrafter):
    def __init__(self):
        super().__init__(spec_num_tokens=4, spec_num_steps=3)
        self.depth: int | None = None

    def configure_algorithm(self, algorithm_config):
        self.depth = int(algorithm_config["depth"])

    def run(self, base_ctx, logits_output, output_tokens, accept_lengths):
        raise NotImplementedError

    def draft(self, *args, **kwargs):
        raise NotImplementedError


class _PlainDrafter(BaseDrafter):
    def run(self, base_ctx, logits_output, output_tokens, accept_lengths):
        raise NotImplementedError

    def draft(self, *args, **kwargs):
        raise NotImplementedError


def test_algorithm_config_reaches_a_plugin_algorithm_unparsed():
    args = _args(speculative_algorithm_config='{"depth": 6, "mode": "tree"}')
    assert json.loads(args.speculative_algorithm_config) == {
        "depth": 6,
        "mode": "tree",
    }


def test_algorithm_config_from_the_cli():
    args = _cli(
        [
            "--sampling-backend",
            "greedy",
            "--speculative-algorithm",
            PLUGIN_ALGORITHM,
            "--speculative-draft-model-path",
            "draft",
            "--speculative-algorithm-config",
            '{"depth": 6}',
        ]
    )
    assert json.loads(args.speculative_algorithm_config) == {"depth": 6}


def test_method_config_in_speculative_config_sets_the_algorithm_config():
    args = _args(
        speculative_algorithm=None,
        speculative_draft_model_path=None,
        speculative_config=json.dumps(
            {
                "method": PLUGIN_ALGORITHM,
                "model": "draft",
                "method_config": {"depth": 6},
            }
        ),
    )
    assert args.speculative_algorithm == PLUGIN_ALGORITHM
    assert json.loads(args.speculative_algorithm_config) == {"depth": 6}
    # The same settings given twice agree.
    _args(
        speculative_algorithm_config='{"depth": 6}',
        speculative_config='{"method_config": {"depth": 6}}',
    )


def test_conflicting_method_config_is_refused():
    with pytest.raises(ValueError, match="conflicts with method_config"):
        _args(
            speculative_algorithm_config='{"depth": 6}',
            speculative_config='{"method_config": {"depth": 8}}',
        )


@pytest.mark.parametrize(
    "algorithm", ["EAGLE3", "MTP", "DFLASH", "DSPARK", "DUMMY", None]
)
def test_algorithm_config_is_refused_without_a_plugin_algorithm(algorithm):
    with pytest.raises(ValueError, match="plugin-registered"):
        _args(
            speculative_algorithm=algorithm,
            speculative_algorithm_config='{"depth": 6}',
        )


@pytest.mark.parametrize("text", ["[1, 2]", "6", "{depth: 6}"])
def test_algorithm_config_must_be_a_json_object(text):
    with pytest.raises(ValueError, match="--speculative-algorithm-config must be"):
        _args(speculative_algorithm_config=text)


def test_a_drafter_reads_its_algorithm_config():
    drafter = _ConfiguredDrafter()
    drafter.configure_algorithm({"depth": 6})
    assert drafter.depth == 6


def test_a_drafter_without_settings_refuses_a_config():
    drafter = _PlainDrafter(spec_num_tokens=4, spec_num_steps=3)
    with pytest.raises(ValueError, match="takes no --speculative-algorithm-config"):
        drafter.configure_algorithm({"depth": 6})


class _AuxTarget(torch.nn.Module):
    """A target that records the layers it is asked to capture."""

    def __init__(self):
        super().__init__()
        self.captured_layers: list[int] | None = None

    def set_eagle3_layers_to_capture(self, layer_ids=None):
        self.captured_layers = list(layer_ids)


def _wire(algorithm: str, *, eagle3_layers_to_capture=None, target=None):
    """Run the real draft-target wiring; the runners stand in for loaded weights."""
    target = _AuxTarget() if target is None else target
    draft_config = SimpleNamespace(
        hf_config={"eagle_aux_hidden_state_layer_ids": [1, 5, 9]},
        requires_request_token_history=False,
    )
    configure_draft_target(
        SimpleNamespace(
            speculative_algorithm=algorithm,
            eagle3_layers_to_capture=eagle3_layers_to_capture,
        ),
        SimpleNamespace(model=target),
        SimpleNamespace(model=torch.nn.Module(), model_config=draft_config),
    )
    return target


@pytest.fixture
def plugin_algorithm():
    yield PLUGIN_ALGORITHM
    drafter_registry._PLUGIN_DRAFTERS.pop(PLUGIN_ALGORITHM, None)


def test_a_plugin_algorithm_can_capture_aux_hidden_states(plugin_algorithm):
    registry.register_drafter(
        plugin_algorithm, _PlainDrafter, captures_aux_hidden_states=True
    )
    assert drafter_registry.drafter_captures_aux_hidden_states(plugin_algorithm)
    assert _wire(plugin_algorithm).captured_layers == [1, 5, 9]
    assert _wire(plugin_algorithm, eagle3_layers_to_capture=[2, 4]).captured_layers == [
        2,
        4,
    ]


def test_a_plugin_algorithm_captures_nothing_by_default(plugin_algorithm):
    registry.register_drafter(plugin_algorithm, _PlainDrafter)
    assert not drafter_registry.drafter_captures_aux_hidden_states(plugin_algorithm)
    assert _wire(plugin_algorithm).captured_layers is None


def test_in_tree_aux_capture_is_eagle3_only():
    assert drafter_registry.drafter_captures_aux_hidden_states("EAGLE3")
    for algorithm in ("MTP", "DFLASH", "DSPARK", "DUMMY", None):
        assert not drafter_registry.drafter_captures_aux_hidden_states(algorithm)


def test_aux_capture_unwinds_with_a_failed_plugin(plugin_algorithm):
    with pytest.raises(RuntimeError, match="boom"):
        with registry.recording():
            registry.register_drafter(
                plugin_algorithm, _PlainDrafter, captures_aux_hidden_states=True
            )
            raise RuntimeError("boom")
    assert not drafter_registry.drafter_captures_aux_hidden_states(plugin_algorithm)


def test_aux_capture_override_replaces_and_rolls_back_the_contract(plugin_algorithm):
    registry.register_drafter(
        plugin_algorithm, _PlainDrafter, captures_aux_hidden_states=True
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        with registry.recording():
            registry.register_drafter(plugin_algorithm, _PlainDrafter, override=True)
            assert _wire(plugin_algorithm).captured_layers is None
            raise RuntimeError("registration failed")
    assert _wire(plugin_algorithm).captured_layers == [1, 5, 9]
    registry.register_drafter(plugin_algorithm, _PlainDrafter, override=True)
    assert _wire(plugin_algorithm).captured_layers is None


def test_scoped_drafter_cannot_enable_capture_for_other_models(plugin_algorithm):
    registry.register_drafter(plugin_algorithm, _PlainDrafter)
    with pytest.raises(ValueError, match="algorithm default"):
        registry.register_drafter(
            plugin_algorithm,
            _PlainDrafter,
            model_cls=torch.nn.Linear,
            captures_aux_hidden_states=True,
        )
    assert _wire(plugin_algorithm).captured_layers is None


def test_scoped_drafter_inherits_capture_and_prevents_contract_change(plugin_algorithm):
    registry.register_drafter(
        plugin_algorithm, _PlainDrafter, captures_aux_hidden_states=True
    )
    registry.register_drafter(
        plugin_algorithm, _PlainDrafter, model_cls=torch.nn.Linear
    )
    assert _wire(plugin_algorithm).captured_layers == [1, 5, 9]
    with pytest.raises(ValueError, match="cannot change auxiliary capture"):
        registry.register_drafter(plugin_algorithm, _PlainDrafter, override=True)
    assert _wire(plugin_algorithm).captured_layers == [1, 5, 9]


def test_native_eagle3_scoped_override_keeps_aux_capture():
    with pytest.raises(RuntimeError, match="undo native override"):
        with registry.recording():
            registry.register_drafter(
                "EAGLE3", _PlainDrafter, model_cls=torch.nn.Linear
            )
            registry.register_drafter("EAGLE3", _PlainDrafter, override=True)
            assert _wire("EAGLE3").captured_layers == [1, 5, 9]
            raise RuntimeError("undo native override")
    assert drafter_registry.drafter_captures_aux_hidden_states("EAGLE3")


def test_capture_plugin_refuses_a_target_without_aux_capture(plugin_algorithm):
    registry.register_drafter(
        plugin_algorithm, _PlainDrafter, captures_aux_hidden_states=True
    )
    with pytest.raises(TypeError, match="AuxHiddenStateCaptureTarget"):
        _wire(plugin_algorithm, target=torch.nn.Linear(1, 1))


def test_plugin_without_capture_accepts_an_ordinary_target(plugin_algorithm):
    registry.register_drafter(plugin_algorithm, _PlainDrafter)
    target = torch.nn.Linear(1, 1)
    assert _wire(plugin_algorithm, target=target) is target


@pytest.mark.parametrize("method_config", [None, [], 1, "depth"])
def test_method_config_requires_an_object(method_config):
    with pytest.raises(ValueError, match="method_config.*JSON object"):
        _args(speculative_config=json.dumps({"method_config": method_config}))


def test_a_plugin_algorithm_may_draft_trees():
    args = _args(
        speculative_eagle_topk=4,
        speculative_num_steps=4,
        speculative_num_draft_tokens=16,
    )
    assert args.speculative_eagle_topk == 4
    assert args.speculative_num_draft_tokens == 16


def test_plugin_lane_depth_is_independent_of_native_tree_sorting():
    args = _args(
        speculative_eagle_topk=2,
        speculative_num_steps=12,
        speculative_num_draft_tokens=8,
    )
    assert args.speculative_draft_headroom_tokens == 22
    assert args.speculative_num_draft_tokens == 8


def test_a_plugin_tree_keeps_the_engine_limits():
    # The lane mask, not the verify window, bounds draft scratch capacity.
    with pytest.raises(ValueError, match="lane slots"):
        _args(
            speculative_eagle_topk=4,
            speculative_num_steps=18,
            speculative_num_draft_tokens=16,
        )
    with pytest.raises(ValueError, match="must be in"):
        _args(
            speculative_eagle_topk=8,
            speculative_num_steps=9,
            speculative_num_draft_tokens=80,
        )


@pytest.mark.parametrize("algorithm", ["DFLASH", "DSPARK", "DUMMY"])
def test_in_tree_algorithms_without_tree_drafting_stay_refused(algorithm):
    with pytest.raises(ValueError, match="tree drafting"):
        _args(
            speculative_algorithm=algorithm,
            speculative_draft_model_path=None,
            speculative_eagle_topk=4,
            # Block drafters need steps == nodes - 1; the tree check refuses anyway.
            speculative_num_steps=15,
            speculative_num_draft_tokens=16,
        )


def test_a_drafter_that_cannot_draft_trees_refuses_the_tree():
    drafter = _PlainDrafter(spec_num_tokens=16, spec_num_steps=4)
    with pytest.raises(NotImplementedError, match="cannot draft trees"):
        drafter.bind_tree(None)


def test_plugin_chain_takes_the_tree_layout_when_declared(plugin_algorithm):
    args = _args(
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )
    assert not args.speculative_draft_tree
    registry.register_drafter(plugin_algorithm, _PlainDrafter, uses_tree_layout=True)
    assert args.speculative_draft_tree
    args._validate_tree_speculation()


def test_tree_layout_rolls_back_with_a_failed_plugin(plugin_algorithm):
    registry.register_drafter(plugin_algorithm, _PlainDrafter)
    with pytest.raises(RuntimeError, match="boom"):
        with registry.recording():
            registry.register_drafter(
                plugin_algorithm, _PlainDrafter, uses_tree_layout=True, override=True
            )
            assert drafter_registry.drafter_uses_tree_layout(plugin_algorithm)
            raise RuntimeError("boom")
    assert not drafter_registry.drafter_uses_tree_layout(plugin_algorithm)
    assert not drafter_registry.drafter_uses_tree_layout(None)


def test_tree_layout_override_replaces_the_contract(plugin_algorithm):
    registry.register_drafter(plugin_algorithm, _PlainDrafter, uses_tree_layout=True)
    registry.register_drafter(plugin_algorithm, _PlainDrafter, override=True)
    assert not drafter_registry.drafter_uses_tree_layout(plugin_algorithm)


def test_scoped_drafter_inherits_the_tree_layout(plugin_algorithm):
    registry.register_drafter(plugin_algorithm, _PlainDrafter)
    with pytest.raises(ValueError, match="algorithm default"):
        registry.register_drafter(
            plugin_algorithm,
            _PlainDrafter,
            model_cls=torch.nn.Linear,
            uses_tree_layout=True,
        )
    registry.register_drafter(
        plugin_algorithm, _PlainDrafter, uses_tree_layout=True, override=True
    )
    registry.register_drafter(
        plugin_algorithm, _PlainDrafter, model_cls=torch.nn.Linear
    )
    assert drafter_registry.drafter_uses_tree_layout(plugin_algorithm)
    with pytest.raises(ValueError, match="cannot change the tree layout"):
        registry.register_drafter(plugin_algorithm, _PlainDrafter, override=True)
    assert drafter_registry.drafter_uses_tree_layout(plugin_algorithm)


def test_plugin_chain_still_rejects_unsupported_tree_contracts(plugin_algorithm):
    registry.register_drafter(plugin_algorithm, _PlainDrafter, uses_tree_layout=True)
    args = _args(
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )
    args.enable_mixed_batch = True
    with pytest.raises(ValueError, match="mixed batches"):
        args._validate_tree_speculation()
