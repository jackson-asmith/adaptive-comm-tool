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
