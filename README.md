# WorldCheck

A harness for checking whether a stateful tool simulator, an LLM "world model" used as an RL
training environment, actually preserves the information an agent needs to be judged correctly.

First target: the public AppWorld adapter in `patronus-ai/mdlm_world_modeling`.

## Run it

Needs Python 3.10+ and three pure-python packages. No GPU, no model weights, no ms-swift install.

```sh
uv venv && uv pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

The first run shallow-fetches the two upstream repos it pins into `.upstream/` (about 100MB) and
takes a few seconds. Set `WORLDCHECK_UPSTREAM` to point the cache somewhere else.

```sh
PYTHONPATH=. .venv/bin/python -m worldcheck.repro          # the observation-level findings
PYTHONPATH=. .venv/bin/python -m worldcheck.reachability   # how much of their RL split is affected
PYTHONPATH=. .venv/bin/python -m worldcheck.reward         # what their reward function scores
```

Findings land in `results/`. The proposed upstream fix is `patches/appworld_plugin_pagination.patch`.
