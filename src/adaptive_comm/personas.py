"""Loading and validating persona definitions."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError


class Persona(BaseModel):
    """Someone who might read a message: what they do, and what they value and dislike."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    description: str = ""
    values: list[str]
    dislikes: list[str]


class PersonaFileError(ValueError):
    """Raised when a personas file is missing, malformed, or empty."""


def _parse(raw: object, source: str) -> list[Persona]:
    if not isinstance(raw, dict) or not raw:
        raise PersonaFileError(f"{source}: expected a mapping of persona name -> definition")

    personas = []
    for name, fields in raw.items():
        if not isinstance(fields, dict):
            raise PersonaFileError(f"{source}: persona '{name}' must be a mapping")
        try:
            personas.append(Persona(name=str(name), **fields))
        except ValidationError as e:
            raise PersonaFileError(f"{source}: persona '{name}' is invalid:\n{e}") from e
    return personas


def load_personas(path: str | Path | None = None) -> list[Persona]:
    """Load personas from a YAML file, or the bundled defaults when path is None."""
    if path is None:
        text = resources.files("adaptive_comm").joinpath("personas.yaml").read_text()
        source = "built-in personas"
    else:
        path = Path(path)
        if not path.is_file():
            raise PersonaFileError(f"personas file not found: {path}")
        text = path.read_text()
        source = str(path)

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PersonaFileError(f"{source}: invalid YAML: {e}") from e
    return _parse(raw, source)
