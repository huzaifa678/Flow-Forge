"""Optimized prompt model."""
from typing import Optional

from pydantic import BaseModel


class OptimizedPrompt(BaseModel):
    """Model representing an optimized prompt from LangChain optimization."""

    original_prompt: str
    optimized_prompt: str
    optimization_technique: str
    confidence_score: float = 0.0
    metadata: Optional[dict] = None