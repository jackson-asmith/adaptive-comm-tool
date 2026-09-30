"""Simulate how a message lands with different personas, and suggest rewrites for each."""

from importlib.metadata import PackageNotFoundError, version

from adaptive_comm.analyzer import Analyzer, MessageAnalysis, PersonaReaction
from adaptive_comm.personas import Persona, load_personas

try:
    __version__ = version("adaptive-comm-tool")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0+unknown"

__all__ = ["Analyzer", "MessageAnalysis", "Persona", "PersonaReaction", "__version__", "load_personas"]
