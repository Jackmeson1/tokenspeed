# Speculative plugin contract fixture

This installable package declares a real `tokenspeed.plugins` entry point.
It registers a neutral `BaseDrafter` consumer that validates `token_offset`
and uses it when selecting a token from CPU auxiliary features.

`test_spec_plugin_fixture.py` installs a copy into a temporary directory with
`pip --no-deps --no-build-isolation --no-index`; it never installs into the
active environment. The tests use real package discovery, server argument
parsing, drafter resolution, and target capture wiring. Tiny CPU modules stand
in for loaded model weights.

The fixture does not implement serving rounds. Its tests do not establish
ModelExecutor initialization order, GPU execution, attention correctness,
CUDA graph behavior, scheduler admission, or end-to-end model quality.
It requires the `drafter.algorithm_config` and `drafter.aux_hidden_capture`
plugin capabilities and does not request tree drafting or sampling.
