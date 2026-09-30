"""Simulate how a message lands with different personas, and suggest rewrites for each."""

from adaptive_comm.analyzer import Analyzer, MessageAnalysis, PersonaReaction
from adaptive_comm.personas import Persona, load_personas

__all__ = ["Analyzer", "MessageAnalysis", "Persona", "PersonaReaction", "load_personas"]
__version__ = "0.2.0"
