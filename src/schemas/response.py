"""Response models for FlowForge API."""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool
    message: str
    data: Optional[dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = False
    message: str
    error: Optional[str] = None
    details: Optional[dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DiagramResponse(BaseModel):
    """Response model for a single generated diagram."""

    diagram_type: str
    title: str
    mermaid_code: str
    is_valid: bool
    validation_feedback: Optional[str] = None


class TimelineResponse(BaseModel):
    """Response model for timeline generation results."""

    milestones: list[str]
    parallel_work_streams: list[str]
    dependencies: Optional[list[str]] = None
    gantt_chart: Optional[str] = None
    raw_timetable: str


class PlanResponse(BaseModel):
    """Response model for plan generation results."""

    task_breakdown: str
    resource_allocation: Optional[str] = None
    risk_mitigation: Optional[str] = None
    communication_structure: Optional[str] = None
    qa_procedures: Optional[str] = None
    budget_estimation: Optional[str] = None
    raw_plan: str


class WorkflowCompleteResponse(BaseModel):
    """Complete workflow response with all generated artifacts."""

    status: str
    proposal_summary: str
    optimized_prompt: Optional[str] = None
    timeline: Optional[dict[str, Any]] = None
    plan: Optional[str] = None
    diagrams: list[dict[str, Any]] = Field(default_factory=list)
    valid_diagram_count: int = 0
    total_diagram_count: int = 0
    overall_validation: bool = False
    validation_feedback: Optional[str] = None
    current_agent: Optional[str] = None
    error: Optional[str] = None