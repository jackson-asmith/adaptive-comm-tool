import io
import json
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest
from conftest import FakeClient, TTYInput, full_payload, reaction

from adaptive_comm import __version__, cli, clipboard
from adaptive_comm import analyzer as analyzer_module
from adaptive_comm.cli import main, output_path_problem, score_style
from adaptive_comm.clipboard import ClipboardError


@pytest.fixture(autouse=True)
def not_interactive(monkeypatch):
    """Tests run without a terminal unless they say otherwise; stdin is empty."""
    monkeypatch.setattr("adaptive_comm.cli.interactive", lambda: False)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))


def analyzed(capsys):
    return [m["message"] for m in json.loads(capsys.readouterr().out)]


def api_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


# --- Output --------------------------------------------------------------


def test_json_output_and_report_file(capsys, tmp_path):
    out = tmp_path / "report.json"

    assert main(["--json", "-o", str(out), "Ship it."], client=FakeClient(full_payload())) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed[0]["message"] == "Ship it."
    assert json.loads(out.read_text()) == printed


def test_table_output(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")  # wide enough that no cell wraps
    payload = full_payload(tone_tags=["direct"])
    payload["reactions"][0] = reaction("pragmatic_engineer", 9, friction=["too short"])
    payload["reactions"][1] = reaction("empathic_pm", 2)

    assert main(["Ship it."], client=FakeClient(payload)) == 0

    out = capsys.readouterr().out
    assert "Ship it." in out and "tone: direct" in out  # short message: tone must not be cut off
    assert "9/10" in out and "2/10" in out and "friction: too short" in out
    assert "rewrite for vision_director" in out


def test_brackets_in_text_are_shown_literally(capsys):
    payload = full_payload(tone_tags=["[bold]"])
    for r in payload["reactions"]:
        r["rewrite"] = "See [link] and [red]x[/red]"

    assert main(["Fix [TODO] now"], client=FakeClient(payload)) == 0

    out = capsys.readouterr().out
    assert "[TODO]" in out and "[link]" in out and "[red]x[/red]" in out and "[bold]" in out


@pytest.mark.parametrize("score, style", [(10, "bold green"), (8, "bold green"), (5, "yellow"), (4, "bold red")])
def test_score_style(score, style):
    assert score_style(score) == style


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"adaptive-comm {__version__}"


def test_list_personas(capsys):
    assert main(["--list-personas", "--only", "empathic_pm"]) == 0
    out = capsys.readouterr().out
    assert "empathic_pm - Product manager" in out and "vision_director" not in out


# --- Personas and flags --------------------------------------------------


def test_only_filters_personas():
    client = FakeClient({"tone_tags": [], "reactions": [reaction("empathic_pm")]})

    assert main(["--only", "empathic_pm", "--json", "hello"], client=client) == 0
    prompt = client.calls[0]["messages"][0]["content"]
    assert '"empathic_pm"' in prompt and "vision_director" not in prompt


def test_unknown_persona_rejected():
    assert main(["--only", "nobody", "hello"], client=FakeClient()) == 2


def test_bad_personas_file(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("- not a mapping")
    assert main(["-p", str(f), "hello"], client=FakeClient()) == 2


@pytest.mark.parametrize(
    "argv, message",
    [
        (["-y", "hello"], "-y/--yes only applies with -c"),
        (["--each-line", "a", "b"], "--each-line applies to --file"),
        (["-c", "-f", "x.txt"], "not allowed with argument"),
    ],
)
def test_ignored_or_conflicting_flags_are_rejected(capsys, argv, message):
    with pytest.raises(SystemExit) as exit_info:
        main(argv, client=FakeClient())
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# --- Input ---------------------------------------------------------------


def test_requires_a_message(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("   \n"))
    assert main([], client=FakeClient()) == 2


def test_file_is_one_message(tmp_path, capsys):
    f = tmp_path / "post.txt"
    f.write_text("First paragraph.\n\nSecond paragraph (with parens) and it's got an apostrophe!\n")

    assert main(["--json", "-f", str(f)], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["First paragraph.\n\nSecond paragraph (with parens) and it's got an apostrophe!"]


def test_each_line_splits(tmp_path, capsys):
    f = tmp_path / "msgs.txt"
    f.write_text("first\n\nsecond\n")

    assert main(["--json", "--each-line", "-f", str(f)], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["first", "second"]


def test_file_dash_reads_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin"))
    assert main(["--json", "-f", "-"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["from stdin"]


@pytest.mark.parametrize(
    "setup, message",
    [
        (lambda d: d / "missing.txt", "cannot read"),
        (lambda d: d, "cannot read"),  # a directory
        (lambda d: (d / "bin.dat", (d / "bin.dat").write_bytes(b"\xca\xfe\xba\xbe"))[0], "not a UTF-8 text file"),
    ],
)
def test_unreadable_file(tmp_path, capsys, setup, message):
    client = FakeClient(full_payload())

    assert main(["-f", str(setup(tmp_path))], client=client) == 2
    assert message in capsys.readouterr().err
    assert client.calls == []


def test_reads_piped_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("Line one.\nLine two.\n"))

    assert main(["--json"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["Line one.\nLine two."]


def test_args_ignore_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("should not be read"))

    assert main(["--json", "from args"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["from args"]


# --- Clipboard and editor ------------------------------------------------


def test_reads_clipboard(monkeypatch, capsys):
    monkeypatch.setattr("adaptive_comm.cli.read_clipboard", lambda: "Para one!\n\nPara (two).\n")

    assert main(["--json", "-c"], client=FakeClient(full_payload())) == 0
    assert analyzed(capsys) == ["Para one!\n\nPara (two)."]


def test_empty_clipboard(monkeypatch, capsys):
    monkeypatch.setattr("adaptive_comm.cli.read_clipboard", lambda: "  \n")
    assert main(["-c"], client=FakeClient()) == 2
    assert "clipboard is empty" in capsys.readouterr().err


def test_clipboard_tool_missing(monkeypatch, capsys):
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: None)
    assert main(["-c"], client=FakeClient()) == 2
    assert "could not read the clipboard" in capsys.readouterr().err


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
    assert code == (0 if sent else 1)
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


@pytest.mark.parametrize("typed, code", [("Typed in the editor", 0), (None, 1)])
def test_no_args_in_terminal_opens_empty_editor(monkeypatch, capsys, typed, code):
    monkeypatch.setattr("adaptive_comm.cli.edit_message", lambda initial="": typed)
    monkeypatch.setattr("sys.stdin", TTYInput(""))

    assert main(["--json"], client=FakeClient(full_payload())) == code
    if typed:
        assert analyzed(capsys) == ["Typed in the editor"]


# --- Checks before any API call ------------------------------------------


@pytest.mark.parametrize(
    "setup, message",
    [
        (lambda d: d / "no" / "such" / "report.json", "does not exist"),
        (lambda d: d, "is a directory"),
    ],
)
def test_bad_output_path_fails_before_any_call(tmp_path, capsys, setup, message):
    client = FakeClient(full_payload())

    assert main(["-o", str(setup(tmp_path)), "a", "b"], client=client) == 2
    assert message in capsys.readouterr().err
    assert client.calls == []


def test_output_path_problem_permissions(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    existing = locked / "report.json"
    existing.write_text("{}")
    existing.chmod(0o444)
    locked.chmod(0o555)
    try:
        assert "not writable" in output_path_problem(str(existing))
        assert "not writable" in output_path_problem(str(locked / "new.json"))
        assert output_path_problem(str(tmp_path / "fine.json")) is None
    finally:
        locked.chmod(0o755)
        existing.chmod(0o644)


def test_report_write_failure_after_run_is_an_error(tmp_path, capsys, monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli.Path, "write_text", refuse)

    assert main(["--json", "-o", str(tmp_path / "r.json"), "hi"], client=FakeClient(full_payload())) == 1
    captured = capsys.readouterr()
    assert "can't write report" in captured.err
    assert json.loads(captured.out)[0]["message"] == "hi"  # results still printed


def test_missing_credentials_fails_before_reading_input(monkeypatch, capsys):
    no_creds = SimpleNamespace(api_key=None, auth_token=None, credentials=None)
    monkeypatch.setattr(analyzer_module.anthropic, "Anthropic", lambda: no_creds)
    monkeypatch.setattr("adaptive_comm.cli.read_clipboard", lambda: pytest.fail("read input first"))

    assert main(["-c"]) == 2
    assert "no Anthropic credentials found" in capsys.readouterr().err


# --- Failures during a run -----------------------------------------------


@pytest.mark.parametrize(
    "error, message",
    [
        (api_error(anthropic.InternalServerError, 529), "API request failed"),
        (api_error(anthropic.AuthenticationError, 401), "authentication failed"),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "API request failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_failure_mid_batch_keeps_completed_results(tmp_path, capsys, error, message):
    out = tmp_path / "report.json"
    client = FakeClient(full_payload(), fail_on=2, error=error)

    code = main(["--json", "-o", str(out), "a", "b", "c", "d"], client=client)

    assert code == 1
    assert len(client.calls) == 3  # stopped at the failure; "d" never sent
    captured = capsys.readouterr()
    assert [m["message"] for m in json.loads(captured.out)] == ["a", "b"]
    assert [m["message"] for m in json.loads(out.read_text())] == ["a", "b"]
    assert message in captured.err and "Stopped after 2 of 4 messages" in captured.err


def test_failure_on_single_message(capsys):
    client = FakeClient(full_payload(), fail_on=0, error=api_error(anthropic.InternalServerError, 500))

    assert main(["only one"], client=client) == 1
    err = capsys.readouterr().err
    assert "API request failed" in err and "Stopped after" not in err


def test_unusable_response_is_skipped_and_run_continues(capsys):
    class OneBad(FakeClient):
        def _create(self, **kwargs):
            response = super()._create(**kwargs)
            if len(self.calls) == 1:
                response.stop_reason = "refusal"
            return response

    code = main(["--json", "bad", "good"], client=OneBad(full_payload()))

    assert code == 1  # not everything succeeded
    captured = capsys.readouterr()
    assert [m["message"] for m in json.loads(captured.out)] == ["good"]
    assert "skipped" in captured.err


# --- Copying rewrites ----------------------------------------------------


@pytest.fixture
def copied(monkeypatch):
    clipboard_contents = []
    monkeypatch.setattr("adaptive_comm.cli.write_clipboard", clipboard_contents.append)
    monkeypatch.setattr("adaptive_comm.cli.interactive", lambda: True)
    return clipboard_contents


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


def test_copy_prompt_ends_on_ctrl_d(monkeypatch, copied):
    monkeypatch.setattr("sys.stdin", TTYInput(""))  # EOF straight away
    assert main(["hello"], client=FakeClient(full_payload())) == 0
    assert copied == []


def test_copy_failure_is_reported(monkeypatch, capsys):
    def broken(_):
        raise ClipboardError("no clipboard tool found")

    monkeypatch.setattr("adaptive_comm.cli.write_clipboard", broken)
    monkeypatch.setattr("adaptive_comm.cli.interactive", lambda: True)
    monkeypatch.setattr("sys.stdin", TTYInput("1\n"))

    assert main(["hello"], client=FakeClient(full_payload())) == 0
    assert "could not copy" in capsys.readouterr().err


def test_no_copy_prompt_when_not_interactive(capsys):
    assert main(["hello"], client=FakeClient(full_payload())) == 0
    assert "Copy a rewrite?" not in capsys.readouterr().err


def test_no_copy_prompt_with_json(copied):
    assert main(["--json", "hello"], client=FakeClient(full_payload())) == 0
    assert copied == []


def test_interactive_needs_terminal_on_both_ends(monkeypatch):
    monkeypatch.undo()  # drop the autouse stand-in
    monkeypatch.setattr("sys.stdin", TTYInput(""))
    monkeypatch.setattr("sys.stdout", io.StringIO())
    assert cli.interactive() is False
    monkeypatch.setattr("sys.stdout", TTYInput(""))
    assert cli.interactive() is True
