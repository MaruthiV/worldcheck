import ast
import importlib
import os
import logging
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PATRONUS_REPO = "https://github.com/patronus-ai/mdlm_world_modeling.git"
PATRONUS_COMMIT = "58e6fe0c963ee4256f6bb02ee99f6b47dc3feb2e"

MSSWIFT_REPO = "https://github.com/modelscope/ms-swift.git"
MSSWIFT_COMMIT = "43b5d8e3d81493b30959d8ea2dc4c1ddb777e308"

CACHE = Path(os.environ.get("WORLDCHECK_UPSTREAM")
            or Path(__file__).resolve().parent.parent / ".upstream")


def _git(args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def ensure_checkout(repo, commit, name):
    dest = CACHE / name
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        _git(["init", "-q", "."], cwd=dest)
        _git(["remote", "add", "origin", repo], cwd=dest)
    if not _has_head(dest) or _git(["rev-parse", "HEAD"], cwd=dest) != commit:
        _git(["fetch", "-q", "--depth", "1", "origin", commit], cwd=dest)
        _git(["checkout", "-q", commit], cwd=dest)
    head = _git(["rev-parse", "HEAD"], cwd=dest)
    if head != commit:
        raise RuntimeError(f"{name}: expected {commit}, got {head}")
    dirty = _git(["status", "--porcelain"], cwd=dest)
    if dirty:
        raise RuntimeError(f"{name}: checkout has local modifications\n{dirty}")
    return dest


def _has_head(dest):
    try:
        _git(["rev-parse", "HEAD"], cwd=dest)
        return True
    except subprocess.CalledProcessError:
        return False


def _classes(path):
    tree = ast.parse(Path(path).read_text())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            out[node.name] = {
                "bases": [ast.unparse(b) for b in node.bases],
                "fields": [t.target.id for t in node.body
                           if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)],
                "node": node,
            }
    return out


def real_request_fields(msswift):
    c = _classes(msswift / "swift/infer_engine/protocol.py")
    return {
        "RolloutInferRequest": sorted(set(c["InferRequest"]["fields"]) | set(c["RolloutInferRequest"]["fields"])),
        "ChatCompletionResponseChoice": sorted(c["ChatCompletionResponseChoice"]["fields"]),
        "ChatMessage": sorted(c["ChatMessage"]["fields"]),
    }


def real_step_convention(msswift):
    path = msswift / "swift/rollout/multi_turn.py"
    tree = ast.parse(path.read_text())
    c = _classes(path)

    sig = None
    for f in c["MultiTurnScheduler"]["node"].body:
        if isinstance(f, ast.FunctionDef) and f.name == "step":
            sig = [a.arg for a in f.args.args]

    init_attrs = []
    for f in c["RolloutScheduler"]["node"].body:
        if isinstance(f, ast.FunctionDef) and f.name == "__init__":
            for stmt in ast.walk(f):
                if isinstance(stmt, ast.Assign):
                    for t in stmt.targets:
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                            init_attrs.append(t.attr)

    calls, feeds = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "step":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                calls.append([ast.unparse(a) for a in node.args])
        if isinstance(node, ast.Subscript) and ast.unparse(node).endswith("['infer_request']"):
            feeds.append(ast.unparse(node))

    return {
        "step_signature": sig,
        "base_init_attrs": init_attrs,
        "self_step_call_args": calls,
        "feed_forward_exprs": sorted(set(feeds)),
    }


@dataclass
class ChatMessage:
    role: str
    content: Any = None
    tool_calls: Optional[List[Any]] = None
    reasoning_content: Optional[str] = None


@dataclass
class ChatCompletionResponseChoice:
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict] = None
    token_ids: Optional[List[int]] = None
    routed_experts: Optional[Any] = None


@dataclass
class RolloutInferRequest:
    messages: List[Dict] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    audios: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    tools: Optional[List[Any]] = None
    objects: Dict[str, Any] = field(default_factory=dict)
    data_dict: Dict = field(default_factory=dict)
    uuid: Optional[str] = None


class MultiTurnScheduler:
    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
        self.infer_engine = infer_engine
        self._tokenizer = kwargs.get("tokenizer", None)
        self.max_turns = max_turns


class ORM:
    pass


SHIMMED = {
    "swift.infer_engine.protocol": ["ChatCompletionResponseChoice", "RolloutInferRequest"],
    "swift.rewards": ["ORM", "orms"],
    "swift.rollout.multi_turn": ["MultiTurnScheduler", "multi_turns"],
    "swift.utils": ["get_logger"],
}


def install_swift_shims():
    for name in ["swift", "swift.infer_engine", "swift.infer_engine.protocol",
                 "swift.rewards", "swift.rollout", "swift.rollout.multi_turn", "swift.utils"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    p = sys.modules["swift.infer_engine.protocol"]
    p.ChatCompletionResponseChoice = ChatCompletionResponseChoice
    p.RolloutInferRequest = RolloutInferRequest
    sys.modules["swift.rewards"].ORM = ORM
    sys.modules["swift.rewards"].orms = {}
    sys.modules["swift.rollout.multi_turn"].MultiTurnScheduler = MultiTurnScheduler
    sys.modules["swift.rollout.multi_turn"].multi_turns = {}
    sys.modules["swift.utils"].get_logger = lambda: logging.getLogger("worldcheck")


def check_shim_fidelity(msswift):
    real = real_request_fields(msswift)
    conv = real_step_convention(msswift)
    ours = {
        "RolloutInferRequest": sorted(RolloutInferRequest.__dataclass_fields__),
        "ChatCompletionResponseChoice": sorted(ChatCompletionResponseChoice.__dataclass_fields__),
        "ChatMessage": sorted(ChatMessage.__dataclass_fields__),
    }
    problems = []
    for cls, fields in real.items():
        if ours[cls] != fields:
            problems.append(f"{cls}: ours={ours[cls]} real={fields}")

    base = ["infer_engine", "_tokenizer", "max_turns"]
    if conv["base_init_attrs"] != base:
        problems.append(f"base __init__ attrs: ours={base} real={conv['base_init_attrs']}")
    if conv["step_signature"] != ["self", "infer_request", "response_choice", "current_turn"]:
        problems.append(f"step signature: real={conv['step_signature']}")
    expected_call = ["current_request", "response_choice", "current_turn"]
    if not conv["self_step_call_args"] or any(a != expected_call for a in conv["self_step_call_args"]):
        problems.append(f"step call args: real={conv['self_step_call_args']}")
    if not conv["feed_forward_exprs"]:
        problems.append("no ret['infer_request'] feed-forward found in real source")

    return {"ok": not problems, "problems": problems,
            "real_fields": real, "convention": {k: v for k, v in conv.items()}}


ADAPTER_MODULES = ["appworld_plugin", "appworld_wm_prompt", "appworld_prompt",
                   "fix_tool_names_and_schemas"]

_loaded_dirs = []


def load_adapter(patronus_or_dir):
    d = Path(patronus_or_dir)
    appworld = str(d / "appworld" if (d / "appworld").is_dir() else d)
    # the adapter imports its siblings by bare name, so only one tree may be on the path at a time
    for prev in _loaded_dirs:
        while prev in sys.path:
            sys.path.remove(prev)
    _loaded_dirs.clear()
    for m in ADAPTER_MODULES:
        sys.modules.pop(m, None)
    sys.path.insert(0, appworld)
    _loaded_dirs.append(appworld)
    install_swift_shims()
    return importlib.import_module("appworld_plugin")
