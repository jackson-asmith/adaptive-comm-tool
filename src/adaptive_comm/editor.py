"""A small multi-line editor in the terminal, for reviewing a message before it's sent."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.input import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import Output

TOOLBAR = " Ctrl-S: send   Ctrl-C: cancel   Enter: new line "


def _bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-s")
    @kb.add("escape", "enter")
    def _send(event):
        event.current_buffer.validate_and_handle()

    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(exception=KeyboardInterrupt)

    return kb


def edit_message(initial: str = "", *, input: Input | None = None, output: Output | None = None) -> str | None:
    """Let the user edit text in place. Returns the text, or None if they cancel.

    input/output are only for tests; normally prompt_toolkit uses the terminal.
    """
    session: PromptSession[str] = PromptSession(
        multiline=True,
        key_bindings=_bindings(),
        bottom_toolbar=TOOLBAR,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "",
        input=input,
        output=output,
    )
    try:
        text = session.prompt("", default=initial)
    except (KeyboardInterrupt, EOFError):
        return None
    return text
