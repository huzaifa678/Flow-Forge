"""Pydantic schemas for FlowForge request and response validation."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DiagramType(str, Enum):
    """Supported diagram types for generation."""

    WORKFLOW = "workflow"
    CI_CD = "ci_cd"
    SYSTEM_DESIGN = "system_design"
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    GANTT = "gantt"


class AudienceType(str, Enum):
    """Target audience for generated documents and diagrams."""

    ENGINEER = "engineer"
    STAKEHOLDER = "stakeholder"


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
    audience_type: AudienceType = Field(
        default=AudienceType.ENGINEER,
        description="Target audience: engineer (technical) or stakeholder (business)",
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
        default=5, ge=1, le=500, description="Estimated team size"
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

    @model_validator(mode="before")
    @classmethod
    def flatten_aliases(cls, data: Any) -> Any:
        """
        Accept both the canonical nested format (proposal + prompt) and a
        flat format where the user posts fields directly at the top level.

        Flat keys recognised for proposal: Project Title * / Project Description *
        / Requirements / Constraints / Timeline (weeks) / Team Size /
        Technology Stack (comma-separated) / Budget Range

        Flat keys recognised for prompt: User Prompt * / Diagram Types / Priority
        """
        if not isinstance(data, dict):
            return data

        flat_proposal_keys = {
            "project title *": "title",
            "project description *": "description",
            "requirements": "requirements",
            "constraints": "constraints",
            "timeline (weeks)": "timeline_weeks",
            "team size": "team_size",
            "technology stack (comma-separated)": "tech_stack",
            "budget range": "budget_range",
        }

        flat_prompt_keys = {
            "user prompt *": "user_prompt",
            "diagram types": "diagram_types",
            "priority": "priority",
            "optimize prompt": "optimize_prompt",
            "include gantt": "include_gantt",
            "include parallel work": "include_parallel_work",
            "audience type": "audience_type",
        }

        lower_keys = {k.strip().lower(): k for k in data.keys()}

        # Build proposal dict from flat keys if flat fields exist
        proposal_fields: Dict[str, Any] = {}
        for flat_key, target_key in flat_proposal_keys.items():
            if flat_key in lower_keys:
                original_key = lower_keys[flat_key]
                proposal_fields[target_key] = data.pop(original_key)

        # Build prompt dict from flat keys if flat fields exist
        prompt_fields: Dict[str, Any] = {}
        for flat_key, target_key in flat_prompt_keys.items():
            if flat_key in lower_keys:
                original_key = lower_keys[flat_key]
                prompt_fields[target_key] = data.pop(original_key)

        if proposal_fields or prompt_fields:
            # Merge with any existing nested dicts (nested format takes precedence)
            existing_proposal = data.get("proposal", {})
            if isinstance(existing_proposal, dict):
                existing_proposal.update(proposal_fields)
                proposal_fields = existing_proposal

            existing_prompt = data.get("prompt", {})
            if isinstance(existing_prompt, dict):
                existing_prompt.update(prompt_fields)
                prompt_fields = existing_prompt

            if proposal_fields:
                data.setdefault("proposal", proposal_fields)
            if prompt_fields:
                data.setdefault("prompt", prompt_fields)

        return data

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
    