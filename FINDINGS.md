# A pagination defect in the AppWorld adapter, and what the reward does with it

Findings against [`patronus-ai/mdlm_world_modeling`](https://github.com/patronus-ai/mdlm_world_modeling)
at commit `58e6fe0c963ee4256f6bb02ee99f6b47dc3feb2e`, which is HEAD of `main`.

**The claim.** The local responder paginates correctly. The training plugin then applies the page
offset a second time, so every request past page 0 returns an empty list while `total` still reports
the full count. Because the agent is told to page until the response is empty, a sweep halts on the
first page it should have received. On the 80-song library in row `692c77d_2` the agent sees **20 of
80 records** on a task that asks it to act on all of them. Across the RL split, **34 of 34 rows** list
more records than the recommended page size. The reward function scores the truncated trajectory
**identically** to the complete one.

**Verify it in one command**, on three pure-python packages, no GPU and no model weights:

```sh
uv venv && uv pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q      # 17 tests, ~7s from a cold clone
```

This concerns the training path only. Your README states the real environment is used only for
evaluation, never during training, so reported evaluation numbers are unaffected by everything below.

## Mechanism

| Step | Location | Behaviour |
|---|---|---|
| 1 | `appworld_wm_prompt.py:313-315` | The local responder pages **correctly**: `start = page_index * page_limit` over the matched records. |
| 2 | `appworld_plugin.py:421-422` | The plugin applies `start = page_index * page_limit` **again**, to a list that is already one page. |
| 3 | `appworld_prompt.py:36` | "If a task requires all matching records, increment page_index until the response is empty." |
| 4 | `appworld_plugin.py:583-584` | A response counts as success when its text is non-empty and contains no `"error"`. |

The guard is on by default (`appworld_plugin.py:372`, `os.environ.get("APPWORLD_WM_GUARD", "1")`) and
no training script sets the variable, so this is the shipped configuration.

## What the agent receives

Row `692c77d_2`, an 80-record library at the `page_limit=20` your agent prompt recommends:

| Page | Plugin returns | Responder computed | `total` reported |
|---|---|---|---|
| 0 | 20 records | 20 records | 80 |
| 1 | **0** | 20 records | 80 |
| 2 | **0** | 20 records | 80 |
| 3 | **0** | 20 records | 80 |

The agent stops at page 1 under its own stop rule, having seen 20 of 80.

Two results rule out the explanations you would reach for first:

- **It is not cache staleness.** A first-ever request for `page_index=1`, with nothing cached, is also
  empty and still reports `total: 80`. The offset is applied unconditionally.
- **It is not only loss, it is mislabelling.** Request page 1 first and you get nothing; the following
  page 0 request then returns page 1's records (ids `217, 317, 95, ...`) where page 0's
  (`311, 36, 12, ...`) belong.

## Scope

All **34 of 34** rows in `appworld/data/appworld_rl_split_clean.jsonl` list more records than
`page_limit=20` in at least one collection, so every row can reach the defect. Row 0's instruction is
"Give a 1-star rating to all songs in my Spotify song library which I have not liked."

## The reward cannot see it

Scored with your unmodified `AppWorldReward`:

| Trajectory | Records seen | Reward |
|---|---|---|
| Defective sweep | 20 of 80 | **0.0105** |
| Repaired sweep | 80 of 80 | **0.0105** |
| All four pages empty | 0 | 0.0105 |
| One full page | 20 | 0.0105 |
| One empty page with `total: 80` | 0 | 0.0090 |
| One explicit `{"error": ...}` | n/a | 0.0000 |

The gap between the defective and repaired arms is **exactly 0.0000**, and it is structural rather
than incidental: `{"total": 80, "songs": []}` is non-empty and contains no `"error"`, so it increments
`num_success` and `success_rate` stays 1.0 however many records came back. A trajectory that saw
nothing scores the same as one that saw a full page. Only explicit errors are penalised.

## The ablation your README flags

`README.md` annotates the local responder `(optional - worth ablation)`. Three arms, 30 rows with an
enumerable collection larger than `page_limit`, fixed scripted sweeps:

| Arm | Fully enumerated | Mean records seen | Mean reward | Rows halting by page 1 |
|---|---|---|---|---|
| Guard on, shipped | **0 / 30** | 20.0 of 60.4 | 0.01215 | **30** |
| Guard on, patched | **30 / 30** | 60.4 of 60.4 | 0.01192 | **0** |
| Guard off, obedient stub | **0 / 30** | 20.0 of 60.4 | 0.01215 | **30** |

Three things follow. In the shipped configuration not one row can be fully enumerated. **Turning the
guard off changes nothing**, byte-identical on records seen and reward on every row, because the
plugin rewrites the response whatever its origin, so debugging this by toggling the documented flag
leads nowhere. And the patch fixes it completely while the reward goes slightly **down**: per row the
patched arm is never higher, lower on 3 rows and identical on 27.

## Two smaller defects in the same block

- **Single-record responses lose a field.** The loop takes the first non-empty list field of *any*
  dict response. `spotify__show_playlist` for playlist 300 returns `songs` with 8 entries from the
  responder and **5** from the plugin, truncated to the default `page_limit`, with a fabricated
  `total` added. `song_ids` is untouched, so only the first list field is affected.
- **The two layers disagree on the window.** `page_limit=0` yields 0 records from the plugin against 1
  from the responder; `page_index=-1` yields 0 against 20. The plugin lacks the `max(0, ...)` and
  `max(1, ...)` clamps that `_page_args` applies.

## Not a defect: read-after-write

Appendix G documents that mutations are answered deterministically without calling the world model,
and that reads are computed from the baked system prompt. The consequence is visible but it follows
from the design, so it is recorded here rather than reported: updating note 2704 returns `success`
and the next read still returns "Book Reading Lists"; updating note `999999`, which does not exist,
returns `success`; deleting it returns `success` and it remains readable. Flagged only because it
bounds what the simulator can verify within an episode.

## A separate finding, not a plugin bug

Section headers read `(N total, M shown)` and only M records follow. **16 of the 34 rows advertise
more than they list, withholding 1,992 records**; the worst advertises 169 and lists 30. No pagination
fix recovers these, because the simulator was never given them, yet `total` keeps telling the agent
they exist. This is in how the rows were built and is a different claim from everything above.

## The fix

`patches/appworld_plugin_pagination.patch`, 37 lines removed and 14 added. It takes `page_index` and
`page_limit` from `_page_args` so both layers agree, binds the collection key from
`RETURN_KEY_BY_SECTION` instead of guessing the first list field, re-pages **only** when the reply is
longer than one page, and drops `_list_caches`, which stored a single page as if it were the whole
collection and was never evicted.

The re-page branch is kept deliberately. It is the defence against a world model that returns more
than one page, which is Table 7 (i). Two plausible fixes are wrong: deleting the block reintroduces
exactly that failure, and adding `page_index` to the cache key changes nothing because the cached
value is already one page. `tests/test_pagination.py::test_patched_still_pages_an_overlong_reply`
pins the first case.

## Measured versus inferred

Measured: the observation corruption, its reachability across the split, the reward's blindness, and
the ablation. All reproduce offline from your published rows.

Inferred, and stated as such:

- **Effect on a trained agent.** Not measured. It needs a GPU and a served world model. Two factors
  cut the other way and belong in any estimate: `max_turns` is 8 in the data and 10 on the CLI, and
  the training reward is structural with no live-environment task success, so trajectories are short.
  What is demonstrated is corrupted observations and a reward that cannot distinguish them.
- **The guard-off arm uses a stub**, not a neural model. It returns exactly the records the prompt
  declares, which isolates the wrapper's contribution. Whether a real world model obeys the prompt's
  pagination directives is untested here.
- **Nothing here concerns the hosted Digital World Model.** These findings are about this repository.

## Reproducing

```sh
PYTHONPATH=. .venv/bin/python -m worldcheck.repro          # sweep, out-of-order, clamps, truncation
PYTHONPATH=. .venv/bin/python -m worldcheck.reachability   # how much of the split is affected
PYTHONPATH=. .venv/bin/python -m worldcheck.reward         # what AppWorldReward scores
PYTHONPATH=. .venv/bin/python -m worldcheck.ablation       # the three-arm guard ablation
```

Output lands in `results/`. Runs execute your unmodified adapter at the pinned commit against your own
published rows, with real `json_repair` and `requests`. No world model is called; `call_world_model` is
replaced with a function that raises, so the run fails if one is reached. The only substituted
dependency is ms-swift's four imported symbols, and `tests/test_shim_fidelity.py` checks that
substitution against ms-swift at commit `43b5d8e3d81493b30959d8ea2dc4c1ddb777e308`: the dataclass
field lists, the `step` signature, the base `__init__` attributes, the call arguments at every
`self.step(...)` site and the `ret['infer_request']` feed-forward are all parsed from the real source
and asserted. ms-swift is never imported, only parsed, which is why none of its dependencies are
needed.
