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

class ProposalRequest(BaseModel):
    """Model for the client proposal portion of the request."""

    title: str = Field(..., min_length=3, max_length=200, description="Project title")
    description: str = Field(
        ..., min_length=10, description="Detailed project description"
    )
    requirements: List[str] = Field(
        default_factory=list, description="List of project requirements"
    )
    constraints: Optional[List[str]] = Field(
        default_factory=list, description="List of project constraints"
    )
    timeline_weeks: Optional[int] = Field(
        default=12, ge=1, le=104, description="Estimated timeline in weeks"
    )
    team_size: Optional[int] = Field(
        default=5, ge=1, le=50, description="Estimated team size"
    )
    tech_stack: Optional[List[str]] = Field(
        default_factory=list, description="Preferred technology stack"
    )
    budget_range: Optional[str] = Field(
        default=None, description="Budget range (e.g. $50k-$100k)"
    )

    @field_validator("requirements")
    @classmethod
    def validate_requirements(cls, v: List[str]) -> List[str]:
        if v and any(len(req.strip()) < 3 for req in v):
            raise ValueError("Each requirement must be at least 3 characters")
        return v

class DiagramGenerationRequest(BaseModel):
    """Combined request model for diagram generation workflow."""

    proposal: ProposalRequest = Field(..., description="Client proposal details")
    prompt: PromptRequest = Field(..., description="Prompt and generation instructions")
    hf_token: str = Field(..., description="HuggingFace API token")

    @field_validator("hf_token")
    @classmethod
    def validate_hf_token(cls, v: str) -> str:
        if not v or len(v.strip()) < 10:
            raise ValueError("Invalid HuggingFace token")
        return v.strip()


class DiagramResponse(BaseModel):
    """Response model for a single generated diagram."""

    diagram_type: DiagramType
    mermaid_code: str
    title: str
    description: str
    is_valid: bool
    validation_feedback: Optional[str] = None
    image_data: Optional[str] = None


class TimelineOutput(BaseModel):
    """Response model for timeline generation."""

    milestones: List[str]
    parallel_work_streams: List[str]
    dependencies: List[str]
    gantt_chart: str
    raw_timetable: str


class PlanOutput(BaseModel):
    """Response model for plan generation."""

    task_breakdown: str
    resource_allocation: str
    risk_mitigation: str
    communication_structure: str
    qa_procedures: str
    budget_estimation: Optional[str] = None
    raw_plan: str


class WorkflowResponse(BaseModel):
    """Complete workflow response model."""

    status: str = Field(description="Overall workflow status: success | partial | failed")
    proposal_summary: str
    optimized_prompt: Optional[str] = None
    timeline: Optional[TimelineOutput] = None
    plan: Optional[PlanOutput] = None
    diagrams: List[DiagramResponse] = Field(default_factory=list)
    overall_validation: bool = Field(default=False)
    feedback: Optional[str] = None
    error: Optional[str] = None
    current_agent: Optional[str] = None
    