import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import modal

PATRONUS_REPO = "https://github.com/patronus-ai/mdlm_world_modeling.git"
PATRONUS_COMMIT = "58e6fe0c963ee4256f6bb02ee99f6b47dc3feb2e"
MSSWIFT_REPO = "https://github.com/modelscope/ms-swift.git"
MSSWIFT_COMMIT = "43b5d8e3d81493b30959d8ea2dc4c1ddb777e308"
# contemporaneous with their ms-swift commit (2026-04-08), vllm 0.19.0 pins torch 2.10.0 and transformers<5
PINS = ["torch==2.10.0", "vllm==0.19.0", "transformers==4.57.6", "trl==0.29.1"]

AGENT = "LiquidAI/LFM2.5-1.2B-Instruct"
# stand-in for their unreleased SDAR world model, served under the name their wm_proxy.py asks for
WM = "Qwen/Qwen3-4B-Instruct-2507"
WM_ALIAS = "ANONYMOUS/SDAR_world_model_v2"
STOP = ["<|im_end|>", "<|endoftext|>", "<|tool_call_end|>"]
MAX_TURNS = 10

GPU = "A100-80GB"
CPU, MEM_MB = 8.0, 49152
# modal list prices on 2026-09-30: a100-80gb + 8 cores + 48 GiB
USD_PER_SEC = 0.000694 + CPU * 0.0000131 + (MEM_MB / 1024) * 0.00000222
CAP_USD = 50.0

AW = Path("/opt/mdlm/appworld")
RL_ROWS = AW / "data" / "appworld_rl_split_clean.jsonl"
SFT_ROWS = AW / "data" / "appworld_sft_gpt_agent.jsonl"
VOL = Path("/vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .uv_pip_install(*PINS, "peft>=0.11,<0.19", "datasets>=3.0,<4.0", "json_repair==0.63.5",
                    "requests", "tensorboard", "openai")
    .run_commands(
        "printf '%s\\n' " + " ".join(PINS) + " > /opt/pins.txt",
        f"git clone {MSSWIFT_REPO} /opt/ms-swift && git -C /opt/ms-swift checkout {MSSWIFT_COMMIT}",
        "python -m pip install -e /opt/ms-swift -c /opt/pins.txt",
        f"git clone {PATRONUS_REPO} /opt/mdlm && git -C /opt/mdlm checkout {PATRONUS_COMMIT}",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "APPWORLD_WM_GUARD": "1"})
    .add_local_python_source("worldcheck")
)

app = modal.App("worldcheck-grpo", image=image)
vol = modal.Volume.from_name("worldcheck-vol", create_if_missing=True)
gpu_fn = dict(gpu=GPU, cpu=CPU, memory=MEM_MB, volumes={"/vol": vol})


def spent():
    return sum(json.loads(p.read_text())["usd"] for p in (VOL / "ledger").glob("*.json"))


@contextmanager
def ledger(job):
    (VOL / "ledger").mkdir(parents=True, exist_ok=True)
    vol.reload()
    if spent() >= CAP_USD:
        raise RuntimeError(f"budget cap ${CAP_USD} reached, refusing to start {job}")
    t0 = time.time()
    try:
        yield
    finally:
        secs = time.time() - t0
        (VOL / "ledger" / f"{job}.json").write_text(json.dumps({"job": job, "seconds": round(secs), "usd": round(secs * USD_PER_SEC, 3)}))
        vol.commit()


def appworld_dir(arm):
    if arm == "shipped":
        return AW
    from worldcheck import fix
    return fix.patched_tree(AW, Path("/tmp/appworld_patched"))


# tee to the app logs too, the volume copy isn't visible until the job commits
def run(cmd, log, cwd=None, env=None):
    with open(log, "a") as f:
        r = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in r.stdout:
            f.write(line)
            print(line, end="", flush=True)
        r.wait()
    if r.returncode != 0:
        tail = Path(log).read_text()[-4000:]
        raise RuntimeError(f"{cmd[:2]} exited {r.returncode}\n{tail}")


def wait(url, timeout):
    import requests
    end = time.time() + timeout
    while time.time() < end:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError(f"{url} never came up")


# wm server first so it claims its slice before the agent's vllm does
@contextmanager
def world_model(log):
    import requests
    procs = [
        subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", WM,
                          "--served-model-name", WM_ALIAS, "--port", "30001",
                          "--gpu-memory-utilization", "0.15", "--max-model-len", "24576"],
                         stdout=open(log, "a"), stderr=subprocess.STDOUT),
    ]
    try:
        wait("http://localhost:30001/health", 1200)
        procs.append(subprocess.Popen([sys.executable, "wm_proxy.py"], cwd=AW,
                                      env={**os.environ, "SGLANG_PORT": "30001", "WM_PORT": "30000"},
                                      stdout=open(log, "a"), stderr=subprocess.STDOUT))
        wait("http://localhost:30000/health", 120)
        probe = requests.post("http://localhost:30000/predict", timeout=120, json={"input": {
            "state": [], "action": [], "system_prompt": 'Reply with {"ok": true}', "max_tokens": 16}}).json()
        if probe.get("error") or not probe.get("generated_text"):
            raise RuntimeError(f"wm stand-in probe failed: {probe}")
        yield "http://localhost:30000/predict"
    finally:
        for p in procs:
            p.terminate()


def last_checkpoint(out):
    cks = sorted(out.rglob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not cks:
        raise RuntimeError(f"no checkpoint under {out}")
    return cks[-1]


def wm_requests(log):
    return Path(log).read_text().count("POST /v1/chat/completions")


@app.function(cpu=2.0, memory=8192, timeout=1800)
def smoke():
    import importlib.metadata as md
    sys.path.insert(0, str(AW))
    os.environ["WM_ENDPOINT"] = "http://localhost:30000/predict"
    import appworld_plugin
    from swift.infer_engine import VllmEngine, RequestConfig
    from swift.infer_engine.protocol import RolloutInferRequest
    from worldcheck import fix
    patched = fix.patched_tree(AW, Path("/tmp/appworld_patched"))
    head = subprocess.run(["git", "-C", "/opt/mdlm", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"versions": {p: md.version(p) for p in ("torch", "vllm", "transformers", "trl", "ms-swift", "peft")},
            "patronus_head": head, "scheduler": appworld_plugin.AppWorldScheduler.__name__,
            "patched_dir": str(patched), "swift_ok": bool(VllmEngine and RequestConfig and RolloutInferRequest)}


@app.function(timeout=3 * 3600, **gpu_fn)
def sft(seed: int = 0):
    out = VOL / "runs" / f"sft-s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    # upstream's lfm25 sft recipe; their sft file for it isn't in the repo, so the gpt-agent demos stand in
    # max_length raised from 4096 so no multi-turn demo gets dropped
    cmd = ["swift", "sft", "--model", AGENT, "--template", "chatml", "--dataset", str(SFT_ROWS),
           "--torch_dtype", "bfloat16", "--tuner_type", "full", "--learning_rate", "2e-5",
           "--num_train_epochs", "3", "--per_device_train_batch_size", "2", "--gradient_accumulation_steps", "4",
           "--max_length", "16384", "--gradient_checkpointing", "true", "--save_steps", "50",
           "--save_total_limit", "2", "--logging_steps", "1", "--lr_scheduler_type", "cosine",
           "--warmup_ratio", "0.1", "--report_to", "tensorboard", "--save_only_model", "true",
           "--seed", str(seed), "--output_dir", str(out)]
    with ledger(f"sft-s{seed}"):
        run(cmd, out / "log.txt")
        ck = last_checkpoint(out)
    return {"checkpoint": str(ck)}


@app.function(timeout=8 * 3600, **gpu_fn)
def grpo(arm: str, seed: int, init: str):
    name = f"grpo-{arm}-s{seed}"
    out = VOL / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    wd = appworld_dir(arm)
    wm_log = out / "wm_server.txt"
    # their run_lfm25_sft_grpo_v2.sh, changed only where marked below
    cmd = ["swift", "rlhf", "--rlhf_type", "grpo", "--model", init, "--template", "chatml",
           "--external_plugins", "appworld_plugin.py", "--reward_funcs", "appworld_reward",
           "--multi_turn_scheduler", "appworld_scheduler", "--max_turns", str(MAX_TURNS),
           "--completion_length_limit_scope", "per_round", "--stop_words", *STOP,
           "--tuner_type", "full", "--use_vllm", "true", "--vllm_mode", "colocate",
           "--vllm_gpu_memory_utilization", "0.35",  # 0.5 upstream, room for the wm stand-in
           "--vllm_max_model_len", "32768", "--sleep_level", "1", "--offload_model", "true",
           "--offload_optimizer", "true", "--torch_dtype", "bfloat16", "--dataset", str(RL_ROWS),
           "--load_from_cache_file", "true", "--split_dataset_ratio", "0", "--max_completion_length", "512",
           "--max_length", "32768", "--num_train_epochs", "2", "--per_device_train_batch_size", "1",
           "--learning_rate", "2e-6", "--max_grad_norm", "0.5", "--gradient_accumulation_steps", "8",
           "--lr_scheduler_type", "constant", "--gradient_checkpointing", "true", "--save_steps", "10",
           "--save_total_limit", "3", "--logging_steps", "1", "--warmup_ratio", "0.05",
           "--dataloader_num_workers", "4", "--dataset_num_proc", "4", "--num_generations", "8",
           "--temperature", "0.9", "--top_p", "0.95", "--num_iterations", "1", "--beta", "0.1",
           "--loss_scale", "default", "--log_completions", "true",
           "--report_to", "tensorboard",  # wandb upstream
           "--save_only_model", "true", "--seed", str(seed), "--output_dir", str(out)]
    with ledger(name), world_model(wm_log) as endpoint:
        # same single-node env their scripts export
        env = {**os.environ, "WM_ENDPOINT": endpoint, "RUNPOD_ENDPOINT_ID": "", "PYTHONPATH": str(wd),
               "TRAJECTORY_LOG": str(out / "trajectories.jsonl"),
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_VISIBLE_DEVICES": "0",
               "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29512", "WORLD_SIZE": "1", "NODE_RANK": "0",
               "NCCL_SOCKET_IFNAME": "lo", "NCCL_IB_DISABLE": "1", "NCCL_NET": "Socket"}
        run(cmd, out / "log.txt", cwd=wd, env=env)
        ck = last_checkpoint(out)
        (out / "provenance.json").write_text(json.dumps({"wm_requests": wm_requests(wm_log)}))
    return {"checkpoint": str(ck), "wm_requests": wm_requests(wm_log)}


def episode_record(plugin, row, sample, messages, final, sources):
    calls = []
    for m in messages:
        if m["role"] == "assistant":
            calls.append(plugin.parse_tool_call(plugin.strip_thinking(m["content"] or "")))
    answers = [str((c.get("arguments") or {}).get("answer", "")) for c in calls
               if c and c.get("name") == "supervisor__complete_task"]
    pages = []
    for c in calls:
        if c and c.get("name") in LIST_TOOLS:
            try:
                pages.append(int((c.get("arguments") or {}).get("page_index", 0) or 0))
            except (TypeError, ValueError):
                pages.append(0)
    reward = plugin.AppWorldReward()([final], ground_truth=[row.get("ground_truth", "")],
                                     messages=[messages], instruction=[row["instruction"]])[0]
    return {"task_id": row["task_id"], "sample": sample, "ground_truth": row.get("ground_truth", ""),
            "answer": answers[-1] if answers else None, "reward": reward, "turns": len(calls),
            "unparsed_turns": sum(c is None for c in calls), "list_pages": pages, "sources": sources,
            "tools": [c.get("name") if c else None for c in calls]}


LIST_TOOLS = {"spotify__show_song_library", "spotify__show_liked_songs", "spotify__search_songs",
              "spotify__show_album_library", "spotify__show_playlist_library", "spotify__show_recommendations",
              "spotify__show_following_artists", "venmo__show_transactions", "venmo__show_social_feed",
              "venmo__show_received_payment_requests", "venmo__show_sent_payment_requests",
              "venmo__search_friends", "file_system__show_directory", "simple_note__search_notes"}


@app.function(timeout=3 * 3600, **gpu_fn)
def evaluate(model: str, arm: str, tag: str, samples: int = 8, seed: int = 0):
    out = VOL / "results"
    out.mkdir(parents=True, exist_ok=True)
    wd = appworld_dir(arm)
    with ledger(f"eval-{tag}-{arm}"), world_model(out / f"{tag}-{arm}-wm.txt") as endpoint:
        os.environ.update(WM_ENDPOINT=endpoint, RUNPOD_ENDPOINT_ID="",
                          TRAJECTORY_LOG=str(out / f"{tag}-{arm}-steps.jsonl"))
        sys.path.insert(0, str(wd))
        os.chdir(wd)
        import appworld_plugin as plugin
        from swift.infer_engine import RequestConfig, VllmEngine
        from swift.infer_engine.protocol import RolloutInferRequest

        # provenance: count model calls so each observation can be tagged guard or model
        hits = [0]
        real = plugin.call_world_model

        def tagged(prompt, state, action):
            hits[0] += 1
            return real(prompt, state, action)
        plugin.call_world_model = tagged

        engine = VllmEngine(model, template_type="chatml", gpu_memory_utilization=0.45,
                            max_model_len=32768, seed=seed)
        cfg = RequestConfig(max_tokens=512, temperature=0.9, top_p=0.95, stop=STOP)
        sched = plugin.AppWorldScheduler(max_turns=MAX_TURNS)
        rows = [json.loads(l) for l in open(RL_ROWS)]
        eps = []
        for row in rows:
            for k in range(samples):
                req = RolloutInferRequest(messages=json.loads(row["messages"]), uuid=f"{row['task_id']}#{k}",
                                          data_dict={x: row.get(x) for x in ("instruction", "ground_truth",
                                                                             "wm_system_prompt", "max_turns")})
                eps.append({"row": row, "k": k, "req": req, "done": False, "final": "", "sources": []})
        # same order as ms-swift's multi-turn loop: append, check_finished, turn cap, step
        for turn in range(1, MAX_TURNS + 1):
            live = [e for e in eps if not e["done"]]
            if not live:
                break
            resps = engine.infer([e["req"] for e in live], cfg, use_tqdm=False)
            for e, resp in zip(live, resps):
                choice = resp.choices[0]
                text = choice.message.content or ""
                e["req"].messages.append({"role": "assistant", "content": text})
                e["final"] = text
                if sched.check_finished(e["req"], choice, turn) or turn >= MAX_TURNS:
                    e["done"] = True
                    continue
                before = hits[0]
                e["req"] = sched.step(e["req"], choice, turn)["infer_request"]
                e["sources"].append("model" if hits[0] > before else "guard")

        records = [episode_record(plugin, e["row"], e["k"], e["req"].messages, e["final"], e["sources"])
                   for e in eps]
        result = {"model": model, "eval_arm": arm, "tag": tag, "samples": samples, "seed": seed,
                  "temperature": 0.9, "max_turns": MAX_TURNS, "wm_stand_in": WM, "episodes": records}
        (out / f"{tag}-{arm}.json").write_text(json.dumps(result))
    return {"tag": tag, "arm": arm, "episodes": len(records)}


@app.function(cpu=1.0, memory=2048, timeout=24 * 3600, volumes={"/vol": vol})
def orchestrate(stage: str, seeds: tuple = (0, 1, 2)):
    vol.reload()
    if stage == "bench":
        s = sft.remote(0)
        g = grpo.remote("shipped", 0, s["checkpoint"])
        return {"sft": s, "grpo": g, "spent_usd": round(spent(), 2)}
    if stage == "train":
        init = str(last_checkpoint(VOL / "runs" / "sft-s0"))
        todo = [(arm, s) for arm in ("shipped", "patched") for s in seeds
                if not list((VOL / "runs" / f"grpo-{arm}-s{s}").rglob("checkpoint-*"))]
        per_run = json.loads((VOL / "ledger" / "grpo-shipped-s0.json").read_text())["usd"]
        if spent() + per_run * len(todo) > CAP_USD:
            raise RuntimeError(f"{len(todo)} runs at ${per_run} each would pass the ${CAP_USD} cap")
        outs = list(grpo.starmap([(arm, s, init) for arm, s in todo], return_exceptions=True))
        return {"ran": todo, "results": [str(o) for o in outs], "spent_usd": round(spent(), 2)}
    if stage == "eval":
        jobs = [(str(last_checkpoint(VOL / "runs" / "sft-s0")), arm, "sft-s0") for arm in ("patched", "shipped")]
        for arm in ("shipped", "patched"):
            for s in seeds:
                ck = str(last_checkpoint(VOL / "runs" / f"grpo-{arm}-s{s}"))
                jobs += [(ck, ev, f"grpo-{arm}-s{s}") for ev in ("patched", "shipped")]
        outs = list(evaluate.starmap(jobs, return_exceptions=True))
        return {"results": [str(o) for o in outs], "spent_usd": round(spent(), 2)}
    raise ValueError(stage)


@app.local_entrypoint()
def main(stage: str = "smoke"):
    if stage == "smoke":
        print(json.dumps(smoke.remote(), indent=2))
    else:
        print(json.dumps(orchestrate.remote(stage), indent=2))
