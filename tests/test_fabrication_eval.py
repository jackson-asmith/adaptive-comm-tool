"""The fabrication eval's runner, exercised end to end with a fake API (no paid calls)."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest
from helpers import full_payload

EVAL_DIR = Path(__file__).resolve().parents[1] / "evals" / "fabrication"
sys.path.insert(0, str(EVAL_DIR))
import run_eval  # noqa: E402

USAGE = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=0, cache_creation_input_tokens=0)


def response(text, model, stop_reason="end_turn"):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(content=content, model=model, stop_reason=stop_reason, usage=USAGE)


class FakeAPI:
    """Answers the app's call (beta.messages) and the judge's (messages) separately."""

    def __init__(
        self,
        verdict,
        app_model="claude-opus-5-5",
        judge_model="claude-sonnet-5-5",
        app_stop="end_turn",
        app_error=None,
        hang=None,
    ):
        self.verdict, self.app_model, self.judge_model = verdict, app_model, judge_model
        self.app_stop, self.app_error, self.hang = app_stop, app_error, hang
        self.app_calls = self.judge_calls = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._app))
        self.messages = SimpleNamespace(create=self._judge)

    def _app(self, **kwargs):
        self.app_calls += 1
        if self.hang:
            self.hang.wait(5)
        if self.app_error:
            raise self.app_error
        return response(json.dumps(full_payload()), self.app_model, self.app_stop)

    def _judge(self, **kwargs):
        self.judge_calls += 1
        return response(json.dumps(self.verdict), self.judge_model)


FAITHFUL = {"additions": [], "keeps_meaning": True, "meaning_notes": "fine"}
INVENTED = {
    "additions": [{"quote": "since 2 a.m.", "kind": "fact", "why": "new time"}],
    "keeps_meaning": True,
    "meaning_notes": "fine",
}


@pytest.fixture
def eval_env(tmp_path, monkeypatch):
    """Point the runner at a temp directory with two public cases and an approved harness."""
    monkeypatch.setattr(run_eval, "FLOW_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "STATE_FILE", tmp_path / "_state.json")
    cases = [
        {"id": "a", "category": "terse request", "source": "variation", "message": "Restart the agent."},
        {"id": "b", "category": "humor", "source": "variation", "message": "What could go wrong."},
    ]
    monkeypatch.setattr(run_eval, "load_cases", lambda: cases)

    def use(api):
        monkeypatch.setattr(run_eval.anthropic, "Anthropic", lambda **kwargs: api)
        return api

    return SimpleNamespace(dir=tmp_path, use=use)


def run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv])
    run_eval.check_harness(approve=True)
    run_eval.main()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_oracle_scores_100_and_records_everything(eval_env, monkeypatch, capsys):
    api = eval_env.use(FakeAPI(FAITHFUL))
    run([], monkeypatch)

    results = rows(eval_env.dir / "baseline" / "results.jsonl")
    assert len(results) == 2 and api.app_calls == 2 and api.judge_calls == 6  # 3 rewrites each
    row = results[0]
    assert row["status"] == "ok" and row["grade"] == {"no_additions": 1.0, "keeps_meaning": 1.0, "additions": 0}
    assert row["model"] == "claude-opus-5-5" and row["judge_model"] == "claude-sonnet-5-5"
    assert row["usage"]["output_tokens"] == 500 and row["judge_usage"]["output_tokens"] == 1500
    trace = json.loads((eval_env.dir / "baseline" / "traces" / f"{row['prompt_id']}_rep0.json").read_text())
    assert [t["role"] for t in trace[:3]] == ["system", "user", "assistant"]
    assert any(t.get("name") == "judge instructions" for t in trace)
    assert "No additions: 100%" in (eval_env.dir / "report.md").read_text()


def test_null_scores_0(eval_env, monkeypatch):
    eval_env.use(FakeAPI(INVENTED))
    run([], monkeypatch)
    grades = [r["grade"] for r in rows(eval_env.dir / "baseline" / "results.jsonl")]
    assert all(g["no_additions"] == 0.0 and g["additions"] == 3 for g in grades)


def test_api_error_goes_to_errors_not_scores(eval_env, monkeypatch):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.InternalServerError("overloaded", response=httpx.Response(529, request=request), body=None)
    eval_env.use(FakeAPI(FAITHFUL, app_error=error))
    run([], monkeypatch)

    assert rows(eval_env.dir / "baseline" / "results.jsonl") == []
    errors = rows(eval_env.dir / "baseline" / "errors.jsonl")
    assert [e["failure_class"] for e in errors] == ["api_error", "api_error"]


def test_refusal_is_recorded_but_not_scored(eval_env, monkeypatch):
    eval_env.use(FakeAPI(FAITHFUL, app_stop="refusal"))
    run([], monkeypatch)
    results = rows(eval_env.dir / "baseline" / "results.jsonl")
    assert {r["status"] for r in results} == {"refused"} and all(r["grade"] == {} for r in results)
    assert "2 refused" in (eval_env.dir / "report.md").read_text()


def test_wrong_serving_model_is_a_failure(eval_env, monkeypatch):
    eval_env.use(FakeAPI(FAITHFUL, app_model="claude-opus-4-8"))  # e.g. a fallback
    run([], monkeypatch)
    errors = rows(eval_env.dir / "baseline" / "errors.jsonl")
    assert {e["failure_class"] for e in errors} == {"served_model_mismatch"}
    assert all(e["usage"]["output_tokens"] == 500 for e in errors)  # billed, so still counted


def test_timeout(eval_env, monkeypatch):
    release = threading.Event()
    eval_env.use(FakeAPI(FAITHFUL, hang=release))
    try:
        run(["--timeout-s", "1", "--only", "a"], monkeypatch)
    finally:
        release.set()
    assert [e["failure_class"] for e in rows(eval_env.dir / "baseline" / "errors.jsonl")] == ["timeout"]


def test_resume_skips_finished_work(eval_env, monkeypatch):
    api = eval_env.use(FakeAPI(FAITHFUL))
    run(["--only", "a"], monkeypatch)
    run(["--reps", "2"], monkeypatch)  # a/rep0 is done; runs a/rep1, b/rep0, b/rep1
    keys = [(r["prompt_id"], r["rep"]) for r in rows(eval_env.dir / "baseline" / "results.jsonl")]
    assert sorted(keys) == [("a", 0), ("a", 1), ("b", 0), ("b", 1)] and api.app_calls == 4


def test_changed_harness_needs_reapproval(eval_env, monkeypatch):
    run_eval.check_harness(approve=True)
    monkeypatch.setattr(run_eval, "harness_sha", lambda: "something else")
    with pytest.raises(SystemExit, match="approve-harness"):
        run_eval.check_harness(approve=False)


def test_judge_check_answer_key_is_well_formed():
    import yaml

    checks = yaml.safe_load((EVAL_DIR / "judge_check.yaml").read_text())
    assert len(checks) >= 10
    for check in checks:
        assert set(check) == {"original", "rewrite", "expect", "note"}
        assert set(check["expect"]) == {"additions", "keeps_meaning"}


def test_public_cases_are_well_formed():
    import yaml

    cases = yaml.safe_load((EVAL_DIR / "cases.yaml").read_text())
    assert len(cases) == 15 and len({c["id"] for c in cases}) == 15
    assert all(c["source"] == "variation" for c in cases)  # real messages stay in the private file


def test_report_refuses_to_mix_grader_versions(eval_env, monkeypatch):
    eval_env.use(FakeAPI(FAITHFUL))
    run(["--only", "a"], monkeypatch)
    monkeypatch.setattr(run_eval, "harness_sha", lambda: "a-newer-grader")
    run_eval.check_harness(approve=True)
    with pytest.raises(SystemExit, match="mixes results"):
        run(["--only", "b"], monkeypatch)


def test_regrade_rejudges_saved_rewrites_without_app_calls(eval_env, monkeypatch):
    first = eval_env.use(FakeAPI(FAITHFUL))
    run([], monkeypatch)
    source = eval_env.dir / "archive" / "old"
    source.parent.mkdir()
    (eval_env.dir / "baseline").rename(source)

    second = eval_env.use(FakeAPI(INVENTED))
    monkeypatch.setattr(run_eval, "harness_sha", lambda: "new-grader")
    run(["--regrade-from", str(source)], monkeypatch)

    assert second.app_calls == 0 and second.judge_calls == 6
    old = {r["prompt_id"]: r for r in rows(source / "results.jsonl")}
    new = rows(eval_env.dir / "baseline" / "results.jsonl")
    assert {r["prompt_id"] for r in new} == set(old) and first.app_calls == 2
    for r in new:
        assert r["grade"]["no_additions"] == 0.0 and r["harness_sha"] == "new-grader"
        assert r["usage"] == old[r["prompt_id"]]["usage"]  # the app call that produced the rewrites
        assert [v["rewrite"] for v in r["meta"]["verdicts"]] == [
            v["rewrite"] for v in old[r["prompt_id"]]["meta"]["verdicts"]
        ]


def test_regrade_refuses_to_overwrite(eval_env, monkeypatch):
    eval_env.use(FakeAPI(FAITHFUL))
    run(["--only", "a"], monkeypatch)
    with pytest.raises(SystemExit, match="already has results"):
        run(["--regrade-from", str(eval_env.dir / "baseline")], monkeypatch)
