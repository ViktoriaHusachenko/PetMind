"""Backward-compatible import path; the agent lives in `app.agent`."""

from ..agent import TOOLS, PetMindAgent

PetlyAgent = PetMindAgent  # legacy name kept for existing imports and tests

__all__ = ["PetMindAgent", "PetlyAgent", "TOOLS"]
