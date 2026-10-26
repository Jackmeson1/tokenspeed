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

"""Installable CPU consumer of speculative plugin configuration and capture."""

from collections.abc import Mapping
from typing import Any

import torch

from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.plugins import PLUGIN_CAPABILITIES, registry

ALGORITHM = "CONTRACT_FIXTURE"


class FixtureDrafter(BaseDrafter):
    """Consume configured offsets and captured features without serving a model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token_offset = 0

    def configure_algorithm(self, algorithm_config: Mapping[str, Any]) -> None:
        if set(algorithm_config) != {"token_offset"}:
            raise ValueError("the fixture requires exactly token_offset")
        offset = algorithm_config["token_offset"]
        if type(offset) is not int or offset < 0:
            raise ValueError("token_offset must be a nonnegative integer")
        self.token_offset = offset

    def draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.device.type != "cpu":
            raise ValueError("the contract fixture only drafts CPU features")
        return hidden_states.argmax(dim=-1) + self.token_offset

    def run(self, base_ctx, logits_output, output_tokens, accept_lengths):
        raise NotImplementedError(
            "the contract fixture does not execute serving rounds"
        )


REQUIRED_CAPABILITIES = frozenset(
    {"drafter.algorithm_config", "drafter.aux_hidden_capture"}
)


def register() -> None:
    missing = REQUIRED_CAPABILITIES - PLUGIN_CAPABILITIES
    if missing:
        raise RuntimeError(
            f"speculative contract fixture requires plugin capabilities {sorted(missing)}"
        )
    registry.register_drafter(
        ALGORITHM, FixtureDrafter, captures_aux_hidden_states=True
    )
