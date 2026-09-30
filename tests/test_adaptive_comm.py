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


class TTYInput(io.StringIO):
    """Fake interactive stdin: reports itself as a terminal and supplies typed answers."""

    def isatty(self):
        return True


@pytest.mark.parametrize("edited, sent", [("Edited text", True), (None, False)])
def test_clipboard_opens_editor(monkeypatch, capsys, edited, sent):
    seen = {}

    def fake_editor(initial=""):
        seen["initial"] = initial
        return edited

    monkeypatch.setattr("adaptive_comm.cli.read_clipboard", lambda: "  From the clipboard\n")
    monkeypatch.setattr("adaptive_comm.cli.edit_message", fake_editor)
    monkeypatch.setattr("sys.stdin", TTYInput(""))
    client = FakeClient(full_payload())

    code = main(["--json", "-c"], client=client)

    assert seen["initial"] == "From the clipboard"
    assert (code == 0) == sent
    if sent:
        assert analyzed(capsys) == ["Edited text"]
    else:
        assert client.calls == []


def test_clipboard_yes_skips_editor(monkeypatch, capsys):
    monkeypatch.setattr("adaptive_comm.cli.read_clipboard", lambda: "As copied")
    monkeypatch.setattr("adaptive_comm.cli.edit_message", lambda initial="": pytest.fail("editor opened"))
    monkeypatch.setattr("sys.stdin", TTYInput(""))

    assert main(["--json", "-c", "-y"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["As copied"]


def test_no_args_in_terminal_opens_empty_editor(monkeypatch, capsys):
    monkeypatch.setattr("adaptive_comm.cli.edit_message", lambda initial="": "Typed in the editor")
    monkeypatch.setattr("sys.stdin", TTYInput(""))

    assert main(["--json"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["Typed in the editor"]


# The real editor, driven with simulated keystrokes.
from prompt_toolkit.input import create_pipe_input  # noqa: E402
from prompt_toolkit.output import DummyOutput  # noqa: E402

from adaptive_comm.editor import edit_message  # noqa: E402

CTRL_S, CTRL_C, ESC, BACKSPACE = "\x13", "\x03", "\x1b", "\x7f"


@pytest.mark.parametrize(
    "initial, keys, expected",
    [
        ("", "Hello\rworld" + CTRL_S, "Hello\nworld"),  # Enter adds a line; Ctrl-S sends
        ("Draft!", BACKSPACE + "?" + CTRL_S, "Draft?"),  # edit prefilled text
        ("Keep (this)", ESC + "\r", "Keep (this)"),  # Esc then Enter also sends
        ("Anything", CTRL_C, None),  # Ctrl-C cancels
    ],
)
def test_editor_keys(initial, keys, expected):
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        assert edit_message(initial, input=pipe, output=DummyOutput()) == expected


# Copying rewrites after the results.


@pytest.fixture
def copied(monkeypatch):
    clipboard = []
    monkeypatch.setattr("adaptive_comm.cli.write_clipboard", clipboard.append)
    monkeypatch.setattr("adaptive_comm.cli.interactive", lambda: True)
    return clipboard


def test_copy_rewrites_by_number(monkeypatch, capsys, copied):
    monkeypatch.setattr("sys.stdin", TTYInput("2\nabc\n9\n3\n\n"))

    assert main(["hello"], client=FakeClient(full_payload())) == 0

    assert copied == ["rewrite for empathic_pm", "rewrite for vision_director"]
    err = capsys.readouterr().err
    assert "Copied empathic_pm's rewrite." in err
    assert err.count("Enter a number from 1 to 3.") == 2  # "abc" and "9"


def test_copy_numbers_continue_across_messages(monkeypatch, capsys, copied):
    monkeypatch.setattr("sys.stdin", TTYInput("4\n\n"))

    assert main(["first", "second"], client=FakeClient(full_payload())) == 0

    assert copied == ["rewrite for pragmatic_engineer"]
    assert "Copied message 2, pragmatic_engineer's rewrite." in capsys.readouterr().err


def test_no_copy_prompt_when_not_interactive(monkeypatch, capsys):
    monkeypatch.setattr("adaptive_comm.cli.write_clipboard", lambda _: pytest.fail("copied"))
    monkeypatch.setattr("adaptive_comm.cli.interactive", lambda: False)

    assert main(["hello"], client=FakeClient(full_payload())) == 0
    assert "Copy a rewrite?" not in capsys.readouterr().err


def test_no_copy_prompt_with_json(monkeypatch, capsys, copied):
    assert main(["--json", "hello"], client=FakeClient(full_payload())) == 0
    assert copied == []


def test_brackets_in_text_are_shown_literally(monkeypatch, capsys):
    monkeypatch.setattr("adaptive_comm.cli.interactive", lambda: False)
    payload = {"tone_tags": ["[bold]"], "reactions": [dict(reaction(p.name), rewrite="See [link] and [red]x[/red]") for p in load_personas()]}

    assert main(["Fix [TODO] now"], client=FakeClient(payload)) == 0
    out = capsys.readouterr().out
    assert "[TODO]" in out and "[link]" in out and "[red]x[/red]" in out and "[bold]" in out
