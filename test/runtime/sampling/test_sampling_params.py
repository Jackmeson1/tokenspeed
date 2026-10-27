# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""``SamplingParams.verify`` range checks.

Each check has to match its own message: an inclusive bound that the message
excludes is how a bad value reaches the penalty kernels and corrupts a
distribution. ``repetition_penalty`` divides positive logits by the penalty, so
``0.0`` would turn every repeated token into ``+inf`` at sample time instead of
being rejected at the request boundary.
"""

import pytest

from tokenspeed.runtime.sampling.sampling_params import SamplingParams


def test_repetition_penalty_rejects_zero():
    params = SamplingParams(repetition_penalty=0.0)
    with pytest.raises(ValueError, match=r"repetition_penalty must be in \(0, 2\]"):
        params.verify(vocab_size=32)


def test_repetition_penalty_rejects_negative():
    params = SamplingParams(repetition_penalty=-0.5)
    with pytest.raises(ValueError, match=r"repetition_penalty must be in \(0, 2\]"):
        params.verify(vocab_size=32)


def test_repetition_penalty_accepts_positive_and_upper_bound():
    SamplingParams(repetition_penalty=1.0).verify(vocab_size=32)
    SamplingParams(repetition_penalty=2.0).verify(vocab_size=32)


def test_repetition_penalty_rejects_above_upper_bound():
    params = SamplingParams(repetition_penalty=2.5)
    with pytest.raises(ValueError, match=r"repetition_penalty must be in \(0, 2\]"):
        params.verify(vocab_size=32)


def test_min_p_stays_inclusive_at_zero():
    # Guards the neighbouring check against a blanket flip of every bound in
    # this file: min_p=0 is valid and must keep passing.
    SamplingParams(min_p=0.0).verify(vocab_size=32)
    SamplingParams(min_p=1.0).verify(vocab_size=32)


# ``min_new_tokens`` is not enforced by any finish check or sampling backend,
# so a positive minimum is only accepted where the request cannot end before
# its length budget.


def test_min_new_tokens_rejected_when_eos_can_end_the_request():
    for params in (
        SamplingParams(min_new_tokens=5),
        SamplingParams(min_new_tokens=5, max_new_tokens=16),
    ):
        with pytest.raises(ValueError, match=r"min_new_tokens=5 is only supported"):
            params.verify(vocab_size=32)


@pytest.mark.parametrize(
    "stop_or_grammar",
    [
        {"stop": "x"},
        {"stop": ""},
        {"stop": ["x", "y"]},
        {"json_schema": "{}"},
        {"regex": "a+"},
        {"ebnf": 'root ::= "a"'},
        {"structural_tag": "{}"},
    ],
)
def test_min_new_tokens_rejected_when_a_stop_or_grammar_can_end_the_request(
    stop_or_grammar,
):
    params = SamplingParams(min_new_tokens=5, ignore_eos=True, **stop_or_grammar)
    with pytest.raises(ValueError, match=r"min_new_tokens=5 is only supported"):
        params.verify(vocab_size=32)
    # Same verdict after normalize(), the order the engine applies them in.
    params.normalize(tokenizer=None)
    with pytest.raises(ValueError, match=r"min_new_tokens=5 is only supported"):
        params.verify(vocab_size=32)


def test_min_new_tokens_accepted_when_only_length_can_end_the_request():
    # The shape of the evalscope perf configs: ignore_eos with min == max.
    SamplingParams(min_new_tokens=1024, max_new_tokens=1024, ignore_eos=True).verify(
        vocab_size=32
    )
    SamplingParams(min_new_tokens=5, ignore_eos=True, stop=[]).verify(vocab_size=32)
    params = SamplingParams(min_new_tokens=5, ignore_eos=True, stop_token_ids={3})
    params.normalize(tokenizer=None)
    params.verify(vocab_size=32)


def test_min_new_tokens_default_and_range_checks_unchanged():
    SamplingParams().verify(vocab_size=32)
    with pytest.raises(ValueError, match=r"min_new_tokens must be in"):
        SamplingParams(min_new_tokens=-1).verify(vocab_size=32)
    with pytest.raises(ValueError, match=r"min_new_tokens must be in"):
        SamplingParams(min_new_tokens=3, max_new_tokens=2, ignore_eos=True).verify(
            vocab_size=32
        )
