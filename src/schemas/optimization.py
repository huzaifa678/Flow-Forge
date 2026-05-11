"""Optimized prompt model."""
from pydantic import BaseModel


class OptimizedPrompt(BaseModel):
    """Model representing an optimized prompt from LangChain optimization."""

    original_prompt: str
    optimized_prompt: str
    optimization_technique: str
    confidence_score: float = 0.0
    metadata: dict | None = None