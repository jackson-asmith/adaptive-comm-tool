import io
import json
from types import SimpleNamespace

import pytest

from adaptive_comm.analyzer import AnalysisError, Analyzer, build_schema
from adaptive_comm.cli import main
from adaptive_comm.personas import PersonaFileError, load_personas


class FakeClient:
    """Stands in for anthropic.Anthropic and returns a canned response."""

    def __init__(self, payload=None, stop_reason="end_turn"):
        self.payload = payload
        self.stop_reason = stop_reason
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = [] if self.payload is None else [SimpleNamespace(type="text", text=json.dumps(self.payload))]
        return SimpleNamespace(stop_reason=self.stop_reason, content=content)


def reaction(name, score=7):
    return {
        "persona": name,
        "rapport_score": score,
        "reaction": f"{name} reaction",
        "friction_points": [],
        "rewrite": f"rewrite for {name}",
    }


@pytest.fixture
def personas():
    return load_personas()


def test_default_personas_load(personas):
    assert [p.name for p in personas] == ["pragmatic_engineer", "empathic_pm", "vision_director"]
    assert all(p.values and p.dislikes for p in personas)


def test_custom_personas_file(tmp_path):
    f = tmp_path / "team.yaml"
    f.write_text("alex:\n  role: Designer\n  values: [craft]\n  dislikes: [rushing]\n")
    [alex] = load_personas(f)
    assert alex.name == "alex" and alex.role == "Designer"


@pytest.mark.parametrize(
    "content",
    ["", "- just a list", "bob: not-a-mapping", "bob:\n  role: X\n  values: [a]\n", "bob:\n  role: X\n  values: [a]\n  dislikes: [b]\n  typo: 1\n"],
)
def test_bad_personas_files_rejected(tmp_path, content):
    f = tmp_path / "bad.yaml"
    f.write_text(content)
    with pytest.raises(PersonaFileError):
        load_personas(f)


def test_schema_pins_persona_names(personas):
    schema = build_schema(personas)
    enum = schema["properties"]["reactions"]["items"]["properties"]["persona"]["enum"]
    assert enum == [p.name for p in personas]


def test_analyze_orders_and_clamps(personas):
    names = [p.name for p in personas]
    payload = {"tone_tags": ["terse"], "reactions": [reaction(names[2]), reaction(names[0], 14), reaction(names[1], -3)]}
    client = FakeClient(payload)

    result = Analyzer(personas, client=client).analyze("Re-run the tests.")

    assert [r.persona for r in result.reactions] == names
    assert [r.rapport_score for r in result.reactions] == [10, 0, 7]
    call = client.calls[0]
    assert "Re-run the tests." in call["messages"][0]["content"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"


def test_missing_persona_is_an_error(personas):
    client = FakeClient({"tone_tags": [], "reactions": [reaction(personas[0].name)]})
    with pytest.raises(AnalysisError, match="missing personas"):
        Analyzer(personas, client=client).analyze("hi")


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_bad_stop_reasons(personas, stop_reason):
    with pytest.raises(AnalysisError):
        Analyzer(personas, client=FakeClient({}, stop_reason=stop_reason)).analyze("hi")


def test_cli_json_output(personas, capsys, tmp_path):
    payload = {"tone_tags": ["direct"], "reactions": [reaction(p.name) for p in personas]}
    out = tmp_path / "report.json"

    code = main(["--json", "-o", str(out), "Ship it."], client=FakeClient(payload))

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed[0]["message"] == "Ship it."
    assert json.loads(out.read_text()) == printed


def test_cli_only_filters_personas(capsys):
    payload = {"tone_tags": [], "reactions": [reaction("empathic_pm")]}
    client = FakeClient(payload)

    assert main(["--only", "empathic_pm", "--json", "hello"], client=client) == 0
    assert '"empathic_pm"' in client.calls[0]["messages"][0]["content"]
    assert "vision_director" not in client.calls[0]["messages"][0]["content"]


def test_cli_rejects_unknown_persona():
    assert main(["--only", "nobody", "hello"], client=FakeClient()) == 2


def full_payload():
    return {"tone_tags": [], "reactions": [reaction(p.name) for p in load_personas()]}


def analyzed(capsys):
    return [m["message"] for m in json.loads(capsys.readouterr().out)]


def test_cli_requires_messages(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("   \n"))
    assert main([], client=FakeClient()) == 2


def test_cli_file_is_one_message(tmp_path, capsys):
    f = tmp_path / "post.txt"
    f.write_text("First paragraph.\n\nSecond paragraph (with parens) and it's got an apostrophe!\n")

    assert main(["--json", "-f", str(f)], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["First paragraph.\n\nSecond paragraph (with parens) and it's got an apostrophe!"]


def test_cli_each_line_splits(tmp_path, capsys):
    f = tmp_path / "msgs.txt"
    f.write_text("first\n\nsecond\n")

    assert main(["--json", "--each-line", "-f", str(f)], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["first", "second"]


def test_cli_reads_piped_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("Line one.\nLine two.\n"))

    assert main(["--json"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["Line one.\nLine two."]


def test_cli_args_ignore_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("should not be read"))

    assert main(["--json", "from args"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["from args"]


def test_cli_reads_clipboard(monkeypatch, capsys):
    monkeypatch.setattr("adaptive_comm.cli.read_clipboard", lambda: "Para one!\n\nPara (two).\n")
    monkeypatch.setattr("sys.stdin", io.StringIO("should not be read"))

    assert main(["--json", "-c"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["Para one!\n\nPara (two)."]


def test_cli_empty_clipboard(monkeypatch):
    monkeypatch.setattr("adaptive_comm.cli.read_clipboard", lambda: "  \n")
    assert main(["-c"], client=FakeClient()) == 2


def test_cli_clipboard_tool_missing(monkeypatch):
    monkeypatch.setattr("adaptive_comm.cli.shutil.which", lambda _: None)
    assert main(["-c"], client=FakeClient()) == 2


def test_cli_clipboard_and_file_conflict():
    with pytest.raises(SystemExit):
        main(["-c", "-f", "x.txt"], client=FakeClient())
