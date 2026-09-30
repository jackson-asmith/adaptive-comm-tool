"""Shared test helpers. No test makes a network call or needs an API key."""

import io
import json
from types import SimpleNamespace

import pytest

from adaptive_comm.personas import load_personas


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def fallback_block():
    return SimpleNamespace(type="fallback")


class FakeClient:
    """Stands in for anthropic.Anthropic.

    Returns `payload` as JSON text, or `content` blocks as given. If `fail_on` is
    set, the call with that index (0-based) raises `error` instead.
    """

    def __init__(self, payload=None, stop_reason="end_turn", content=None, fail_on=None, error=None):
        self.payload = payload
        self.stop_reason = stop_reason
        self.content = content
        self.fail_on = fail_on
        self.error = error
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if self.fail_on == index:
            raise self.error
        if self.content is not None:
            content = self.content
        else:
            content = [] if self.payload is None else [text_block(json.dumps(self.payload))]
        return SimpleNamespace(stop_reason=self.stop_reason, content=content)


class TTYInput(io.StringIO):
    """Fake interactive stdin: reports itself as a terminal and supplies typed answers."""

    def isatty(self):
        return True


def reaction(name, score=7, friction=()):
    return {
        "persona": name,
        "rapport_score": score,
        "reaction": f"{name} reaction",
        "friction_points": list(friction),
        "rewrite": f"rewrite for {name}",
    }


def full_payload(**overrides):
    payload = {"tone_tags": [], "reactions": [reaction(p.name) for p in load_personas()]}
    payload.update(overrides)
    return payload


@pytest.fixture
def personas():
    return load_personas()
