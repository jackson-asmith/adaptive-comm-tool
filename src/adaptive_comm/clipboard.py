"""Read and write the system clipboard using whichever platform tool is installed."""

from __future__ import annotations

import shutil
import subprocess

# Windows PowerShell 5.1 pipes text in the console's legacy code page, which garbles
# anything outside ASCII (curly quotes, dashes, emoji). Force UTF-8, without a byte
# order mark, in both directions. clip.exe is avoided for the same reason.
_UTF8 = "(New-Object System.Text.UTF8Encoding $false)"
_POWERSHELL = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]

# Tried in order; the first one installed wins.
READ_COMMANDS = [
    ["pbpaste"],  # macOS
    ["wl-paste", "--no-newline"],  # Linux, Wayland
    ["xclip", "-selection", "clipboard", "-o"],  # Linux, X11
    ["xsel", "--clipboard", "--output"],  # Linux, X11
    [*_POWERSHELL, f"[Console]::OutputEncoding = {_UTF8}; Get-Clipboard -Raw"],  # Windows, WSL
]

WRITE_COMMANDS = [
    ["pbcopy"],
    ["wl-copy"],
    ["xclip", "-selection", "clipboard", "-i"],
    ["xsel", "--clipboard", "--input"],
    [*_POWERSHELL, f"[Console]::InputEncoding = {_UTF8}; Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
]

NO_TOOL = "no clipboard tool found (install wl-clipboard or xclip on Linux)"


class ClipboardError(RuntimeError):
    """Raised when no clipboard tool is available or running it fails."""


def _run(commands: list[list[str]], input: str | None = None) -> str:
    for cmd in commands:
        if shutil.which(cmd[0]) is None:
            continue
        # Exit status is checked below so the error can include the tool's stderr.
        result = subprocess.run(  # noqa: PLW1510
            cmd, input=input, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise ClipboardError(f"{cmd[0]} failed: {detail}")
        return result.stdout
    raise ClipboardError(NO_TOOL)


def read_clipboard() -> str:
    """The clipboard's text, with Windows line endings normalized to \\n."""
    return _run(READ_COMMANDS).replace("\r\n", "\n")


def write_clipboard(text: str) -> None:
    """Put text on the clipboard."""
    _run(WRITE_COMMANDS, input=text)
