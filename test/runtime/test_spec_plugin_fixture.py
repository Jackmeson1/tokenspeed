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

"""Installed speculative-plugin contracts, exercised with CPU model fixtures."""

import os
import sys

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import importlib.metadata
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime import plugins
from tokenspeed.runtime.execution import drafter as drafter_registry
from tokenspeed.runtime.execution.factory import configure_draft_target
from tokenspeed.runtime.utils.server_args import (
    ServerArgs,
    parse_speculative_algorithm_config,
)


@pytest.fixture(scope="module")
def installed_plugin(tmp_path_factory):
    root = tmp_path_factory.mktemp("spec-plugin")
    source = root / "source"
    shutil.copytree(Path(__file__).parent / "fixtures" / "spec_plugin", source)
    target = root / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--disable-pip-version-check",
            "--target",
            str(target),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return target


@pytest.fixture
def discovered_plugin(installed_plugin, monkeypatch):
    monkeypatch.syspath_prepend(str(installed_plugin))
    monkeypatch.setattr(plugins, "_loaded", False)
    monkeypatch.setattr(plugins, "_loaded_plugins", {})
    monkeypatch.setattr(drafter_registry, "_PLUGIN_DRAFTERS", {})
    # Isolate the installed fixture from unrelated plugins in the test environment.
    other_names = [
        ep.name
        for ep in importlib.metadata.entry_points(group=plugins.ENTRY_POINT_GROUP)
        if ep.name != "spec_contract_fixture"
    ]
    monkeypatch.setenv(plugins.DISABLE_ENV_VAR, ",".join(other_names))
    loaded = plugins.ensure_loaded()
    assert [info.package for info in loaded] == ["tokenspeed-spec-contract-fixture"]
    import tokenspeed_spec_fixture

    return tokenspeed_spec_fixture


class _Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.capture_layers: tuple[int, ...] = ()

    def set_eagle3_layers_to_capture(self, layer_ids):
        self.capture_layers = tuple(layer_ids)

    def forward(self, inputs):
        if not self.capture_layers:
            return None
        return torch.cat(
            [inputs * (layer + 1) for layer in self.capture_layers], dim=-1
        )


@dataclass
class _DraftConfig:
    hf_config: dict = field(
        default_factory=lambda: {"eagle_aux_hidden_state_layer_ids": [0, 2]}
    )
    requires_request_token_history: bool = False


@dataclass
class _Runner:
    # Only loaded-model holders are replaced; configure_draft_target is real.
    model: torch.nn.Module
    model_config: _DraftConfig = field(default_factory=_DraftConfig)


def test_installed_plugin_consumes_configured_auxiliary_features(discovered_plugin):
    args = ServerArgs(
        model="fixture",
        speculative_algorithm=discovered_plugin.ALGORITHM,
        speculative_draft_model_path="draft",
        speculative_algorithm_config='{"token_offset": 7}',
        sampling_backend="greedy",
    )
    target = _Target()
    draft = torch.nn.Identity()
    configure_draft_target(args, _Runner(target), _Runner(draft))
    assert target.capture_layers == (0, 2)
    hidden_states = target(torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(hidden_states, torch.tensor([[1.0, 2.0, 3.0, 6.0]]))
    implementation = drafter_registry.get_drafter_impl(
        args.speculative_algorithm, draft
    )
    drafter = implementation(spec_num_tokens=4, spec_num_steps=3)
    drafter.configure_algorithm(
        parse_speculative_algorithm_config(args.speculative_algorithm_config)
    )
    assert drafter.draft(hidden_states).tolist() == [10]


def test_installed_plugin_honors_explicit_capture_layers(discovered_plugin):
    args = ServerArgs(
        model="fixture",
        speculative_algorithm=discovered_plugin.ALGORITHM,
        speculative_draft_model_path="draft",
        eagle3_layers_to_capture="1",
        sampling_backend="greedy",
    )
    target = _Target()
    configure_draft_target(args, _Runner(target), _Runner(torch.nn.Identity()))
    torch.testing.assert_close(
        target(torch.tensor([[1.0, 2.0]])), torch.tensor([[2.0, 4.0]])
    )


@pytest.mark.parametrize(
    "config", [{"unused": 7}, {"token_offset": -1}, {"token_offset": True}]
)
def test_installed_plugin_rejects_settings_it_cannot_consume(discovered_plugin, config):
    implementation = drafter_registry.get_drafter_impl(
        discovered_plugin.ALGORITHM, torch.nn.Identity()
    )
    drafter = implementation(spec_num_tokens=4, spec_num_steps=3)
    with pytest.raises(ValueError, match="token_offset"):
        drafter.configure_algorithm(config)
