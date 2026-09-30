"""The real editor, driven with simulated keystrokes."""

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from adaptive_comm.editor import edit_message

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
