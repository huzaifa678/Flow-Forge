"""Pydantic schemas for FlowForge request and response validation."""
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class DiagramType(str, Enum):
    """Supported diagram types for generation."""

    WORKFLOW = "workflow"
    CI_CD = "ci_cd"
    SYSTEM_DESIGN = "system_design"
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    GANTT = "gantt"

class PromptRequest(BaseModel):
    """Model for the prompt/custom instructions portion of the request."""

    user_prompt: str = Field(
        ..., min_length=10, description="User's raw prompt/instructions"
    )
    diagram_types: List[DiagramType] = Field(
        default_factory=lambda: [DiagramType.WORKFLOW, DiagramType.CI_CD],
        description="Types of diagrams to generate",
        min_length=1,
    )
    optimize_prompt: bool = Field(
        default=True, description="Whether to optimize the prompt using LangChain"
    )
    include_gantt: bool = Field(
        default=True, description="Whether to include Gantt chart in timeline"
    )
    include_parallel_work: bool = Field(
        default=True, description="Whether to identify parallel work streams"
    )
    priority: str = Field(
        default="medium",
        description="Project priority level",
        pattern=r"^(low|medium|high|critical)$",
    )
