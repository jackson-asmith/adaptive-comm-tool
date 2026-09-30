import os
from types import SimpleNamespace

import pytest

from adaptive_comm import clipboard
from adaptive_comm.clipboard import ClipboardError, read_clipboard, write_clipboard


@pytest.fixture
def fake_tools(monkeypatch):
    """Pretend only the named tools are installed, and record what gets run."""
    state = {"installed": set(), "runs": [], "result": SimpleNamespace(returncode=0, stdout="", stderr="")}
    monkeypatch.setattr(clipboard.shutil, "which", lambda name: name if name in state["installed"] else None)

    def run(cmd, **kwargs):
        state["runs"].append((cmd, kwargs.get("input")))
        return state["result"]

    monkeypatch.setattr(clipboard.subprocess, "run", run)
    return state


def test_read_uses_first_installed_tool(fake_tools):
    fake_tools["installed"] = {"xclip", "xsel"}
    fake_tools["result"].stdout = "copied text"

    assert read_clipboard() == "copied text"
    assert [cmd[0] for cmd, _ in fake_tools["runs"]] == ["xclip"]


def test_write_sends_text_as_input(fake_tools):
    fake_tools["installed"] = {"pbcopy"}

    write_clipboard("hello\n\nworld")
    assert fake_tools["runs"] == [(["pbcopy"], "hello\n\nworld")]


@pytest.mark.parametrize("func, args", [(read_clipboard, ()), (write_clipboard, ("x",))])
def test_no_tool_installed(fake_tools, func, args):
    with pytest.raises(ClipboardError, match="no clipboard tool"):
        func(*args)


@pytest.mark.parametrize("stderr, message", [("Error: no display", "no display"), ("", "exit code 1")])
def test_tool_failure_is_reported(fake_tools, stderr, message):
    fake_tools["installed"] = {"wl-paste"}
    fake_tools["result"] = SimpleNamespace(returncode=1, stdout="", stderr=stderr)

    with pytest.raises(ClipboardError, match=message):
        read_clipboard()


def test_read_normalizes_windows_line_endings(fake_tools):
    fake_tools["installed"] = {"powershell.exe"}
    fake_tools["result"].stdout = "line one\r\nline two\r\n"

    assert read_clipboard() == "line one\nline two\n"


def test_powershell_commands_force_utf8():
    """Windows PowerShell would otherwise use the legacy code page and garble non-ASCII text."""
    read_ps, write_ps = clipboard.READ_COMMANDS[-1], clipboard.WRITE_COMMANDS[-1]
    assert read_ps[0] == write_ps[0] == "powershell.exe"
    assert "-NonInteractive" in read_ps and "-NonInteractive" in write_ps
    assert "OutputEncoding" in read_ps[-1] and "UTF8Encoding $false" in read_ps[-1]
    assert "InputEncoding" in write_ps[-1] and "Set-Clipboard" in write_ps[-1]
    assert not any(cmd[0] == "clip.exe" for cmd in clipboard.WRITE_COMMANDS)


@pytest.mark.skipif(
    not os.environ.get("ADAPTIVE_COMM_CLIPBOARD_TESTS"),
    reason="uses the real clipboard; set ADAPTIVE_COMM_CLIPBOARD_TESTS=1 to run",
)
def test_real_clipboard_round_trip():
    text = "It\u2019s \u201cquoted\u201d \u2014 caf\u00e9, na\u00efve, \U0001f44d\nSecond line (with [brackets])!"
    write_clipboard(text)
    assert read_clipboard().rstrip("\n") == text
