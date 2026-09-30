"""Ask Claude how each persona would react to a message, and how to rephrase it for them."""

from __future__ import annotations

import json
from typing import Any, Literal, get_args

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam
from pydantic import BaseModel, ValidationError, field_validator

from adaptive_comm.personas import Persona

Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: tuple[Effort, ...] = get_args(Effort)

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT: Effort = "medium"

SYSTEM_PROMPT = """\
You are a workplace communication coach. You will be given a message someone \
wants to send to a colleague, plus a set of personas describing people who might \
receive it. For each persona, work out how that person would honestly read the \
message given what they value and dislike.

For every persona:
- rapport_score: an integer from 0 to 10 for how well the message lands with them \
(10 = it builds trust, 5 = neutral or mildly off, 0 = it damages the relationship).
- reaction: one or two sentences in the persona's own voice describing how they \
take it.
- friction_points: the specific words, phrasing, or omissions that cost rapport \
with this persona. Empty if there are none.
- rewrite: the same message rewritten for this persona. Keep the sender's intent, \
facts, and ask intact, match the original's length and register roughly, and do \
not invent details. If the original already lands well, a light touch-up is fine.

Also return tone_tags: two to four short labels for the original message's tone \
(for example "direct", "terse", "appreciative", "defensive").

Personas differ, so their scores should differ when their preferences pull in \
different directions. Judge only the text you are given."""


class PersonaReaction(BaseModel):
    """How one persona reacts to a message, and a rewrite aimed at them."""

    persona: str
    rapport_score: int
    reaction: str
    friction_points: list[str]
    rewrite: str

    @field_validator("rapport_score")
    @classmethod
    def _clamp(cls, v: int) -> int:
        """Keep scores in 0-10 even if the model strays outside the range."""
        return max(0, min(10, v))


class MessageAnalysis(BaseModel):
    """The result for one message: its tone and one reaction per persona, in persona order."""

    message: str
    tone_tags: list[str]
    reactions: list[PersonaReaction]


class AnalysisError(RuntimeError):
    """Raised when Claude declines or returns something unusable."""


class MissingCredentialsError(RuntimeError):
    """Raised when the Anthropic SDK found no API key, token, or credentials profile."""


def has_credentials(client: anthropic.Anthropic) -> bool:
    """Whether the SDK resolved any credentials (env vars, `ant auth login` profile, or federation)."""
    return any(getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials"))


def response_text(content: list[Any]) -> str | None:
    """The answer's text, ignoring anything before a server-side fallback.

    If a safety classifier declines and a fallback model takes over, the content
    holds a `fallback` marker block followed by the fallback model's answer. In
    non-streaming responses the declined model's partial output is omitted, but
    only reading after the last marker keeps this correct either way.
    """
    last_fallback = max((i for i, b in enumerate(content) if b.type == "fallback"), default=-1)
    texts = [b.text for b in content[last_fallback + 1 :] if b.type == "text"]
    return "".join(texts) if texts else None


def build_schema(personas: list[Persona]) -> dict[str, Any]:
    """JSON schema for the structured output, with persona names pinned to an enum."""
    return {
        "type": "object",
        "properties": {
            "tone_tags": {"type": "array", "items": {"type": "string"}},
            "reactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "persona": {"type": "string", "enum": [p.name for p in personas]},
                        "rapport_score": {"type": "integer"},
                        "reaction": {"type": "string"},
                        "friction_points": {"type": "array", "items": {"type": "string"}},
                        "rewrite": {"type": "string"},
                    },
                    "required": ["persona", "rapport_score", "reaction", "friction_points", "rewrite"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["tone_tags", "reactions"],
        "additionalProperties": False,
    }


def build_user_prompt(message: str, personas: list[Persona]) -> str:
    persona_json = json.dumps([p.model_dump() for p in personas], indent=2)
    return f"<personas>\n{persona_json}\n</personas>\n\n<message>\n{message}\n</message>"


class Analyzer:
    """Scores messages against a fixed set of personas using Claude.

    Pass `client` to supply your own `anthropic.Anthropic` (or a test double).
    Without one, a client is created from the environment, and
    MissingCredentialsError is raised if no credentials can be found.
    """

    def __init__(
        self,
        personas: list[Persona],
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        effort: Effort = DEFAULT_EFFORT,
    ) -> None:
        if not personas:
            raise ValueError("at least one persona is required")
        self.personas = personas
        if client is None:
            try:
                client = anthropic.Anthropic()
            except anthropic.CredentialsError as e:  # e.g. ANTHROPIC_PROFILE names a missing profile
                raise MissingCredentialsError(str(e)) from e
            if not has_credentials(client):
                raise MissingCredentialsError("no Anthropic credentials found")
        self.client = client
        self.model = model
        self.effort = effort
        self._schema = build_schema(personas)

    def analyze(self, message: str) -> MessageAnalysis:
        """Analyze one message with one API call.

        Raises AnalysisError if Claude declines or the response is unusable.
        API errors from the SDK (anthropic.APIError) propagate unchanged.
        """
        messages: list[BetaMessageParam] = [
            {"role": "user", "content": build_user_prompt(message, self.personas)},
        ]
        output_config: BetaOutputConfigParam = {
            "effort": self.effort,
            "format": {"type": "json_schema", "schema": self._schema},
        }
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=messages,
            output_config=output_config,
            # If a safety classifier declines, retry server-side on Anthropic's
            # recommended fallback model instead of failing the whole run.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        if response.stop_reason == "refusal":
            raise AnalysisError("Claude declined to analyze this message")
        if response.stop_reason == "max_tokens":
            raise AnalysisError("response was cut off before it finished")

        text = response_text(response.content)
        if text is None:
            raise AnalysisError("response contained no text")

        try:
            data = json.loads(text)
            analysis = MessageAnalysis(message=message, **data)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            raise AnalysisError(f"could not parse response: {e}") from e

        # Keep reactions in persona-file order, one per persona.
        by_name = {r.persona: r for r in analysis.reactions}
        missing = [p.name for p in self.personas if p.name not in by_name]
        if missing:
            raise AnalysisError(f"response is missing personas: {', '.join(missing)}")
        analysis.reactions = [by_name[p.name] for p in self.personas]
        return analysis
