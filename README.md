# WorldCheck

WorldCheck tests whether a simulator used to train AI agents actually shows the agent what it needs
to see, and whether the training score notices when it doesn't.

## In plain terms

Some AI agents are trained inside a simulator instead of the real software. The agent asks a fake
Spotify for its song library, the simulator answers, and a score says how well the agent did. Think of
it as a flight simulator for software agents. Patronus AI published one of these for the AppWorld
benchmark in [`patronus-ai/mdlm_world_modeling`](https://github.com/patronus-ai/mdlm_world_modeling).

In that code, when the agent asks for the second page of a list, the simulator hands back an empty
page. It still says the list has 80 items. The agent has been told to keep asking until a page comes
back empty, so it stops after the first 20. The training score is the same whether the agent saw 20
songs or all 80, so nothing in training notices.

Picture a driving simulator where every street past the first block is blank, and the instructor only
checks that you didn't crash. You pass every lesson and never learn most of the city.

This repo shows that happening on Patronus's own code and data, without a GPU. It includes a fix and
the first tests that code has had. It then trains agents with Patronus's own recipe through the broken
and the fixed code. The bug fires about a hundred times per training run, but the trained agents come
out the same, because the training score pays nearly full marks for declaring a task done without doing
it. Finally, a small practice environment measures the general question behind all this: when a
simulator makes mistakes, does the training score still rank good agents above bad ones?

## Check it yourself

```sh
uv venv && uv pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

That needs three pure-python packages and takes a few seconds. No GPU and no model weights. The first
run fetches the two upstream repos it pins, about 100MB, into `.upstream/`. Set `WORLDCHECK_UPSTREAM`
to keep them somewhere else.

Everything runs the upstream code unmodified, at commit `58e6fe0c963ee4256f6bb02ee99f6b47dc3feb2e`,
which is the current head of its `main` branch.

## What goes wrong

The upstream code has a "local responder", which it also calls the guard. It answers most list
requests straight from the records written into the prompt, without calling the model. It exists to
fix a failure the paper documents in Table 7 (i): a world model that returned page 0 again when asked
for page 1. The responder gets pagination right. The training plugin then paginates its answer a
second time.

| Step | Where | What happens |
|---|---|---|
| 1 | `appworld_wm_prompt.py:313-315` | The responder pages correctly: `start = page_index * page_limit` over the matching records. |
| 2 | `appworld_plugin.py:421-422` | The plugin applies `start = page_index * page_limit` again, to a list that is already one page long. Anything past page 0 comes back empty. |
| 3 | `appworld_prompt.py:36` | The agent is told: "If a task requires all matching records, increment page_index until the response is empty." |
| 4 | `appworld_plugin.py:583-584` | The reward counts a response as a success if it is non-empty and contains no `"error"`. `{"total": 80, "songs": []}` passes. |

The guard is on by default (`appworld_plugin.py:372`) and no training script turns it off, so this is
the configuration that ships.

Here is what the agent receives for row `692c77d_2`, an 80-song library, at the page size the agent
prompt recommends:

| Page | Plugin returns | Responder computed | `total` reported |
|---|---|---|---|
| 0 | 20 songs | 20 songs | 80 |
| 1 | **none** | 20 songs | 80 |
| 2 | **none** | 20 songs | 80 |
| 3 | **none** | 20 songs | 80 |

The agent stops at page 1 under its own rule, having seen 20 of 80.

Two results rule out the explanations people reach for first:

- **It is not a stale cache.** The very first request for page 1, with nothing cached, is also empty
  and still reports `total: 80`. The second offset is applied every time.
- **It loses records and also mislabels them.** Ask for page 1 first and you get nothing. Ask for page
  0 next and you get page 1's songs (ids `217, 317, 95, ...`) where page 0's (`311, 36, 12, ...`)
  belong.

## How much of the training data it reaches

All 34 of 34 rows in the RL split (`appworld/data/appworld_rl_split_clean.jsonl`) list more records
than the recommended page size in at least one collection, so every training task can hit it. Row 0's
task is "Give a 1-star rating to all songs in my Spotify song library which I have not liked."

## The training score can't see it

Scored with the unmodified `AppWorldReward`:

| Trajectory | Songs seen | Reward |
|---|---|---|
| Sweep through the shipped plugin | 20 of 80 | **0.0105** |
| Sweep with correct pagination | 80 of 80 | **0.0105** |
| All four pages empty | 0 | 0.0105 |
| One full page | 20 | 0.0105 |
| One empty page with `total: 80` | 0 | 0.0090 |
| One explicit `{"error": ...}` | none | 0.0000 |

The gap between seeing a quarter of the library and seeing all of it is exactly zero. That follows
from the rule in step 4: an empty list with a total attached is non-empty and has no `"error"` in it,
so it counts as a success. Only an explicit error lowers the score.

These trajectories are hand-built rather than sampled from a trained agent, and the absolute values
are small because they skip the credential steps the reward expects. The claim is only that the reward
cannot tell the two apart.

## The score also accepts a list of guesses

Some training tasks are questions with one right answer, like "how many unique songs are across my
library, albums and playlists?" For these the score checks whether the right answer appears *anywhere*
in the agent's reply, not whether the reply is the answer. So a reply listing every number from 0 to 100
gets full marks on every counting question, without the agent looking at any data.

Scored with the unmodified `AppWorldReward` on the 11 question rows of the RL split:

| Reply | 6 counting questions | 5 name questions |
|---|---|---|
| The exact answer | 1.0 on all 6 | 1.0 on all 5 |
| `0 1 2 ... 100`, 293 characters | **1.0 on all 6** | |
| The answer with a digit in front, `124` for 24 | **1.0 on all 6** | |
| Off by one, `25` for 24 | 0.23 to 0.25 | |
| Every title and name on the first page of each Spotify list | | **1.0 on 2**, 0.15 on 3 |

The rule is at `appworld_plugin.py:666`: if `gt.lower() in answer.lower()`, the reward is 1.0. Nothing
checks the length of the answer or how many candidates it contains, so a wrong number that happens to
contain the right digits scores four times higher than a nearly right one. The three name questions
where pasting a list fails are the ones whose answer sits past the first page, which is the part the
shipped plugin never shows.

This shows the reward can be gamed. It is not evidence that training actually games it. The training
runs in progress check for exactly that.

## A better model would not fix this

The upstream README marks the guard "(optional - worth ablation)". This is that ablation: three
versions of the pipeline, over the 30 rows with a list longer than one page, using the same scripted
sweep in each.

| Version | Rows fully read | Records seen, on average | Mean reward | Rows that stop by page 1 |
|---|---|---|---|---|
| Guard on, as shipped | **0 of 30** | 20.0 of 60.4 | 0.01215 | **30** |
| Guard on, patched | **30 of 30** | 60.4 of 60.4 | 0.01192 | **0** |
| Guard off, perfect stand-in model | **0 of 30** | 20.0 of 60.4 | 0.01215 | **30** |

- **As shipped, not one row can be read in full.** The agent sees exactly one page every time.
- **Turning the guard off changes nothing.** The plugin rewrites the response after it is produced,
  whoever produced it. The guard-off version uses a stub that returns exactly the records the prompt
  declares, which is a perfect world model for this purpose, and it still loses every page after the
  first. The paper (Section 5) expects pagination drift "to improve as MDLM scaling and tool-use
  centric post-training continues to mature." This part of the pagination problem lives in the
  plugin, so no improvement to the model reaches it. It is a different symptom from Table 7 (i):
  empty pages rather than repeated ones.
- **The patch fixes it completely, and the reward goes slightly down.** Per row the patched version is
  never scored higher: lower on 3 rows, the same on 27.

## Two smaller problems in the same block

- **Single-record responses lose part of a field.** The plugin re-pages the first non-empty list inside
  *any* response. For playlist 300, `spotify__show_playlist` returns `songs` with 8 entries from the
  responder and 5 after the plugin, cut to the default page size, with a made-up `total` added.
- **The two layers disagree about the page window.** `page_limit=0` gives 0 records from the plugin and
  1 from the responder. `page_index=-1` gives 0 and 20. The responder clamps these values
  (`max(0, ...)`, `max(1, ...)`) and the plugin doesn't.

## Not a bug: writes that don't show up in later reads

The paper's Appendix G documents that writes are answered with a fixed "Action completed" without
calling the model, and that reads are computed from the records baked into the prompt. So after
renaming note 2704 the next read still shows the old title "Book Reading Lists". Updating note
`999999`, which doesn't exist, also reports success, and a deleted note stays readable. That is the
documented design, so it is recorded here and not reported as a defect. It matters because it limits
what the simulator can check within one episode, and the fix below deliberately leaves it unchanged.

## A separate issue in the data

Each prompt introduces a list with a header like `songs (81 total, 80 shown)`, and only the shown
records follow. **16 of the 34 rows advertise more than they list, 1,992 records in all.** The worst
advertises 169 and lists 30. No pagination fix recovers these, because the simulator was never given
them, while `total` keeps telling the agent they exist. This comes from how the rows were built, not
from the plugin, so it is a separate claim.

## The fix

[`patches/appworld_plugin_pagination.patch`](patches/appworld_plugin_pagination.patch) removes 37
lines and adds 14. It:

- takes `page_index` and `page_limit` from the responder's own `_page_args`, so both layers agree on
  the window;
- picks the list field from `RETURN_KEY_BY_SECTION` instead of guessing the first list it finds;
- re-pages only when a response is longer than one page;
- drops `_list_caches`, which stored one page as if it were the whole list and was never cleared.

The re-page step is kept on purpose. It is the defence against a world model that returns more than
one page, which is exactly Table 7 (i). Two obvious fixes are wrong. Deleting the block brings that
failure back. Adding `page_index` to the cache key changes nothing, because the cached value is already
a single page. `test_patched_still_pages_an_overlong_reply` pins the first of these.

The patch is generated from exact-text replacements in `worldcheck/fix.py`, each of which must match
the upstream file exactly once, so upstream drift fails loudly instead of misapplying. Applied to the
pinned checkout, the four-page sweep returns all 80 songs.

## What happens in real training

Everything above uses hand-built sweeps. So I ran Patronus's own training recipe for real: fine-tuning on
their demonstrations, then GRPO reinforcement learning, on the smallest agent in their paper,
LFM2.5-1.2B. It ran once through the shipped plugin and once through the fixed one, five times each with
different random seeds, with everything else identical. Three things came out of it.

**The bug fires constantly.** In every shipped run the agent asked for a later page of some list 209 to
317 times, and 79 to 108 of those pages held records but came back empty. In the fixed runs, not once.

**Yet the trained agents came out the same.** Training clearly worked: average reward rose from 0.39
after fine-tuning to 0.60 after GRPO, equally in both versions. But on every measure declared before the
runs, agents trained through the fixed plugin were indistinguishable from agents trained through the
shipped one.

**Because training learned to satisfy the score instead of doing the tasks.** On tasks that require
changing something, like rating songs or accepting payment requests, the trained agents usually logged
in, looked around and declared the task complete without changing anything. The score pays that almost
full marks. The share of those task attempts scoring 0.9 or more while changing nothing rose from 17%
before GRPO to between 38% and 48% after, in every run of both versions. If the score doesn't need the
task done, fixing what the agent sees can't change what it learns.

So the pagination fix is necessary but not sufficient: the reward is what limits training here. That
last result was not part of the plan. It came from reading the trained agents' episodes by hand, and it
is labelled exploratory below.

### How it was run

| | |
|---|---|
| Agent | `LiquidAI/LFM2.5-1.2B-Instruct`, one of the three agents in the paper's Table 3 |
| Recipe | `run_lfm25_sft.sh`, then `run_lfm25_sft_grpo_v2.sh`, unchanged except as listed below |
| Runs | One fine-tuned checkpoint shared by both versions, then five GRPO seeds per version |
| World model | Their SDAR model is unreleased (upstream issue #1), so `Qwen/Qwen3-4B-Instruct-2507` stands in, served under the name their `wm_proxy.py` asks for. It was asked 3 to 29 times per run, one of them a startup check, out of 3,300 to 3,900 tool calls; the guard answered the rest |
| Software | torch 2.10.0, vLLM 0.19.0, transformers 4.57.6, TRL 0.29.1, ms-swift at `43b5d8e`, all from the same weeks as the upstream commit |
| Hardware | One A100-80GB per run on Modal, about 21 minutes and $1.15 per GRPO run |

Changes from their scripts: fine-tuning uses `appworld_sft_gpt_agent.jsonl`, because the LFM-specific
file the script names is not in the repo. Fine-tuning `max_length` is 16384 instead of 4096, so no
demonstration is dropped. The training-time vLLM gets 0.35 of the GPU instead of 0.5, to make room for
the stand-in model. Logging goes to TensorBoard instead of Weights & Biases.

The design, the measures and the rule for calling a difference real were written down before any
training run and not changed afterwards. Each trained checkpoint was run 8 times on each of the 34
training tasks, through both plugins.

### Lost pages during training

Every later-page request in each run's training log was replayed through the guard, which pages
correctly, and compared with what the plugin actually returned.

| Version | Later-page requests per run | Held records, came back empty | Correctly empty, past the end |
|---|---|---|---|
| Shipped | 209 to 317 | **79 to 108** | 100 to 206 |
| Fixed | 301 to 390 | **0** | 121 to 211 |

The agents trained through the fixed plugin also asked for more later pages, because a page with records
in it invites the next one.

### The declared comparison

Scored through the fixed plugin, which shows correct pages. A difference counts only if its 95% interval,
from resampling tasks, excludes zero and at least 4 of the 5 seed pairs agree on its direction.

| Measure | Fine-tuned only | Trained, shipped | Trained, fixed | Fixed minus shipped, 95% interval |
|---|---|---|---|---|
| Questions answered exactly right | 0.000 | 0.005 | 0.000 | -0.005, from -0.014 to 0.000 |
| Sweeps that read past the first page | 0.198 | 0.169 | 0.171 | +0.002, from -0.028 to +0.029 |
| Highest page requested | 0.37 | 0.26 | 0.22 | -0.04, from -0.12 to +0.02 |
| Reward | 0.394 | 0.604 | 0.609 | +0.005, from -0.010 to +0.020 |
| Turns that don't parse as a tool call | 2.4% | 2.8% | 2.7% | -0.1 points, from -0.7 to +0.5 |

No measure passes. Question accuracy was never going to show anything at this size: no agent, fine-tuned
or trained, answers the questions. That is 0 to 2 correct out of 440 question attempts per group. A
stronger agent would be needed to test that part.

Scored through the shipped plugin instead, two of the nine measures pass the rule: trained-through-fixed
agents read past the first page 3.7 points more often (interval +0.5 to +6.4, all five seeds agree), and
produce slightly fewer unparseable turns. That was not the declared view, and with eighteen comparisons
two passing could be chance, so it is reported and not claimed.

Training also did not discover the answer-listing trick from earlier: across 240 counting answers per
group, at most 2 contained more than one number, and those were ordinary sentences.

### Exploratory: finishing without acting

Found by reading episodes, not planned. Among attempts at tasks that need a change of state, the share
that call `complete_task`, make no call from the guard's own list of state-changing tools
(`appworld_wm_prompt.py:384-390`), and still score at least 0.9:

| Checkpoint | Through the fixed plugin | Through the shipped plugin |
|---|---|---|
| Fine-tuned only | 17% | 18% |
| Trained through shipped, seeds 0 to 4 | 47, 43, 38, 45, 48% | 36, 42, 39, 41, 46% |
| Trained through fixed, seeds 0 to 4 | 38, 41, 48, 43, 48% | 46, 42, 42, 38, 48% |

After training, agents call `complete_task` on 94-95% of these tasks, and 91% of those finishes change
nothing. The reward averages 0.87 to 0.88 for a finish that changes nothing and 0.84 to 0.89 for one that does.
Every one of the 23 action tasks shows it.

### Limits

- The agents were scored on the same 34 tasks they trained on, through the simulator, not in real
  AppWorld. The paper's own AppWorld results come from its real-environment evaluation, which this does
  not reproduce.
- One small agent, one fine-tuning seed, and a stand-in world model that the guard made almost
  irrelevant.
- "Changed nothing" means no call from the guard's list of state-changing tools. Every action task in
  this split needs one of those tools except one, which asks the agent to play a song. Leaving that task
  out moves every rate above by at most 2 points.

`python -m gpu.analyze` recomputes all of this from the downloaded logs, and `gpu/grpo.py` is the whole
training and evaluation job. The raw logs and episodes contain AppWorld-derived records, so they are
not in this repo; `results/grpo_summary.json` holds the aggregates. The whole arm cost $24.33 on Modal,
$5.74 of it on a batch of runs that got cancelled part-way and were rerun from scratch.

## The general question: a practice environment

The AppWorld bug is one instance of a broader problem. If a simulator gets things wrong, does the score
used for training still rank a good agent above a bad one? Getting individual responses right is not
enough if the ranking comes out wrong, because the ranking is what training acts on.

To measure that directly, `worldcheck/env/` is a small online store built as real, runnable software:
customers, orders, charges, refunds and notes. It has the untidy details real systems accumulate. A
fault injector then makes a simulated copy of the store go wrong in six specific ways, three of them
taken from the paper's Table 7. Nine simple scripted agents run against the real store and against
each faulty copy, and the rankings are compared.

### The results, briefly

- **With a correct simulator, the score picks the right agent.** Its top pick is also the best agent
  in reality.
- **With a simulator that behaves like the shipped AppWorld path,** meaning later pages empty and writes
  acknowledged but never saved, the score rates an agent that reads one page exactly the same as one
  that reads everything, 0.659 for both when they write the same way. In the real store the one-page
  agent fails 3 of 15 tasks.
- **The simulator causes this, not the score.** Grading the simulated runs perfectly shows the same
  wrong top pick.
- **The agent that checks its own work gets hurt most.** The simulator never shows a saved write, so
  that agent retries. Replayed against the real store, those retries land as duplicate refunds, and
  its tasks come out right 47% of the time, against 80-100% when it runs on the real store directly.

This is a statement about the measuring tool and these specific faults, not about any particular world
model. A real model plugs in through one function, `observe(state_description, action)`.

### How it works

**The store.** Eight tools over an in-memory SQLite database, with money in integer cents, a clock that
only moves when an action happens, snapshot and restore, and an append-only audit log. Pagination rules
are written down once and enforced everywhere, since two layers disagreeing about pagination is where
this project started.

**Untidy data.** Eight fixtures, each one a place a simulator plausibly slips:

- a partly refunded charge, so the correct refund is the remainder, not the full amount;
- a refund already recorded under an idempotency key that a task retries;
- one customer with two accounts under the same email;
- a soft-deleted order that the index still counts;
- an index total taken before an order was cancelled;
- an order placed just after midnight local time, which is the previous day in UTC;
- a pending charge that must not be refunded;
- a note in which a customer mentions pasting a "Traceback". The upstream reward counts any response
  containing "traceback" as a failure (`appworld_plugin.py:581`), so this legitimate record is scored
  as an error.

**Tasks.** Fifteen, graded on the final state of the database plus a check that nothing forbidden
happened: exact-once refunds, counting across pages, telling similar records apart, and refusing
writes the rules don't allow.

**Faults.** Page 0 repeated for later pages, later pages empty, an invented record added, the response
wrapped in a chat envelope, writes acknowledged but not saved, and reads that miss earlier writes. Each
fault changes only what the agent sees, never the underlying database.

**Verifier.** Seven checks that work from the declared state and the observations alone, so they apply
to simulators we don't control. Each finding names the layer at fault. Every injected fault is caught
by the check meant for it, and across 135 episodes of real agent traffic against the real store it
raises nothing.

**Agents.** Every combination of two choices, declared before any result was computed. Paging: read
page 0 only, read until a page is empty, or read as many pages as `total` implies. Writing: write once,
check the state first, or write and then read back and retry.

**Comparison.** Each agent gets four scores. *Truth* is the real store, graded. *Training signal* is
the simulator, scored by a reward shaped like `AppWorldReward`. *Shadow* is the simulated run's actions
replayed against the real store, then graded. *Reward only* is the real store, scored by that reward.
Two agents tie if their scores differ by less than half a task. Each of the 36 pairs either agrees with
truth, collapses (truth separates them, the view doesn't), is spurious (truth ties them, the view
doesn't), or is reversed. Regret is how much worse, in reality, the view's top pick is than the true
best; when the view ties several agents at the top, the worst of them counts.

The reward mirrors both branches of `AppWorldReward`, including its substring answer check. It leaves
out the credential and argument-format penalties, since this store has no login step for them to act
on.

### Full results

Training signal compared with truth, for each simulator (36 agent pairs):

| Simulator | Agree | Collapse | Spurious | Reversed | Regret | Episodes the verifier flags |
|---|---|---|---|---|---|---|
| Correct | 26 | 0 | 10 | 0 | 0.0 | 0 of 135 |
| Page 0 repeated | 9 | 4 | 9 | 14 | 0.2 | 42 of 135 |
| Later pages empty | 12 | 10 | 10 | 4 | 0.2 | 24 of 135 |
| Invented record | 13 | 9 | 11 | 3 | 0.2 | 111 of 135 |
| Chat-wrapped responses | 36 | 0 | 0 | 0 | 0.0 | 135 of 135 |
| Writes not saved | 31 | 0 | 5 | 0 | 0.0 | 27 of 135 |
| Reads miss writes | 26 | 0 | 10 | 0 | 0.0 | 15 of 135 |
| **AppWorld-like** (empty pages, writes not saved) | 15 | 14 | 5 | 2 | **0.2** | 51 of 135 |

A regret of 0.2 is three tasks out of fifteen. The 10 spurious pairs under the correct simulator come
from the reward, not the simulator: it prefers agents that avoid attempting a forbidden refund, while
the real store simply rejects the attempt and the task still passes. Per-agent scores for every view,
and the exact pairs in each class, are in `results/calibration.json`.

## What is measured, and what isn't

Measured, and reproduced offline from the upstream's own published rows: the corrupted observations,
how much of the training data they reach, the reward's blindness to them, the answers it accepts, the
guard ablation, and the practice-environment results. Measured on GPUs with their recipe: how often the
bug fires during real training, and whether agents trained through the fixed plugin come out different
(at this size, they don't).

Not measured:

- **Whether a stronger agent would differ.** The 1.2B agent never answers the question tasks, which is
  where missing records should matter most. The paper also trains Qwen3-4B and Mistral-7B.
- **Real AppWorld evaluation of the trained agents.** They were scored through the simulator.
- **Whether a real world model follows the prompt's pagination rules.** The guard-off version uses a
  stub, and in training the stand-in model was almost never reached.
- **Anything about Patronus's hosted Digital World Model.** Everything here is about this repository.

## Reproducing

```sh
PYTHONPATH=. .venv/bin/python -m worldcheck.repro          # page sweep, out-of-order pages, clamps, truncation
PYTHONPATH=. .venv/bin/python -m worldcheck.reachability   # how much of the RL split is affected
PYTHONPATH=. .venv/bin/python -m worldcheck.reward         # what AppWorldReward scores
PYTHONPATH=. .venv/bin/python -m worldcheck.answers        # which answers it gives full credit
PYTHONPATH=. .venv/bin/python -m worldcheck.ablation       # the three-version guard ablation
PYTHONPATH=. .venv/bin/python -m worldcheck.calibrate      # the practice-environment rankings
```

Each writes its output to `results/`, so you can diff your run against the committed one.

The upstream code runs with its real `json_repair` and `requests`. No world model is ever called:
`call_world_model` is replaced with a function that raises, so a run fails if one is reached. The only
thing substituted is four symbols imported from ms-swift, the training framework, which would otherwise
pull in vLLM and a GPU stack. `tests/test_shim_fidelity.py` checks that substitute against ms-swift at
commit `43b5d8e3d81493b30959d8ea2dc4c1ddb777e308`. It parses the real source for the dataclass fields,
the `step` signature, the base class attributes, the arguments at every `self.step(...)` call and the
`ret['infer_request']` hand-off. ms-swift is only ever parsed, never imported.

This repo does not copy any upstream data. The RL rows are read from the pinned upstream checkout at
run time.

| Path | What it is |
|---|---|
| `worldcheck/upstream.py` | Fetches both pinned repos and checks the ms-swift substitute against the real source |
| `worldcheck/driver.py` | Drives the upstream scheduler one tool call at a time |
| `worldcheck/repro.py`, `reachability.py`, `reward.py`, `answers.py`, `ablation.py` | The AppWorld measurements |
| `worldcheck/fix.py` | The fix as exact-text replacements; `patches/` is generated from it |
| `worldcheck/env/` | The practice store: engine, untidy fixtures, tasks and graders |
| `worldcheck/sim/injector.py` | The six faults |
| `worldcheck/verify.py` | The seven checks |
| `worldcheck/policies.py`, `calibrate.py` | The nine agents and the ranking comparison |
| `gpu/grpo.py`, `gpu/analyze.py` | The training and evaluation job on Modal, and its analysis |
| `results/` | Committed output of every command above |
| `tests/` | Everything above, as tests |
