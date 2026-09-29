from worldcheck import upstream

EXPECTED_BASE_INIT = ["infer_engine", "_tokenizer", "max_turns"]
EXPECTED_STEP_SIG = ["self", "infer_request", "response_choice", "current_turn"]


def test_shims_match_real_ms_swift(msswift):
    r = upstream.check_shim_fidelity(msswift)
    assert r["ok"], r["problems"]


def test_request_and_response_fields_are_exact(msswift):
    real = upstream.real_request_fields(msswift)
    assert sorted(upstream.RolloutInferRequest.__dataclass_fields__) == real["RolloutInferRequest"]
    assert sorted(upstream.ChatCompletionResponseChoice.__dataclass_fields__) == real["ChatCompletionResponseChoice"]
    assert sorted(upstream.ChatMessage.__dataclass_fields__) == real["ChatMessage"]


def test_we_call_step_the_way_ms_swift_does(msswift):
    c = upstream.real_step_convention(msswift)
    assert c["step_signature"] == EXPECTED_STEP_SIG
    assert c["base_init_attrs"] == EXPECTED_BASE_INIT
    assert c["self_step_call_args"], "no self.step(...) call found upstream"
    for args in c["self_step_call_args"]:
        assert args == ["current_request", "response_choice", "current_turn"]
    assert c["feed_forward_exprs"], "no ret['infer_request'] feed-forward found upstream"


def test_only_ms_swift_symbols_are_shimmed():
    assert set(upstream.SHIMMED) == {
        "swift.infer_engine.protocol", "swift.rewards",
        "swift.rollout.multi_turn", "swift.utils",
    }
