# scripts/debug

Manual inspection and smoke-check utilities collected during development.
**These are not a test suite** — several are named `test_*.py` for historical
reasons, but they execute real work at import (API calls, dataset loads,
environment generation) rather than asserting anything.

Run them individually, from the repository root:

```bash
python scripts/debug/test_resize.py
```

`pyproject.toml` excludes this directory from pytest collection so a bare
`pytest` invocation cannot fire them off by accident.

| Script | What it checks |
| --- | --- |
| `check_hf_solvable.py`, `test_hf_solvability.py`, `test_solvability.py` | solvability labelling of generated levels |
| `test_minigrid_generation.py` | renders XLand-MiniGrid levels for each ruleset and prints the LLM prompts |
| `test_resize.py` | the resize pipeline against the configured environment backend |
| `testing_rendering.py`, `testing_scr.py`, `extract_grid.py` | renderer output and grid extraction |
| `inspect_ds.py`, `inspect_dataset.py` | compiled HuggingFace dataset contents and batch shapes |
| `test_api.py`, `test_file_upload.py` | connectivity to the Open WebUI and Gemini endpoints |
| `test_jax.py`, `testt_envs.py`, `debug_parse.py` | JAX/device setup, environment construction, LLM response parsing |
