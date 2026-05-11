"""FastAPI router for FlowForge diagram generation endpoints."""
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_hf_token
from src.schema.helpers import format_workflow_response
from src.schemas.request import DiagramGenerationRequest
from src.schemas.response import (
    APIResponse,
    WorkflowCompleteResponse,
)
from src.workflow.graph_workflow import run_flowforge_workflow


router = APIRouter(prefix="/api/v1", tags=["FlowForge"])


@router.post(
    "/generate-diagrams",
    response_model=WorkflowCompleteResponse,
    responses={
        200: {"description": "Diagrams generated successfully"},
        400: {"description": "Invalid request data", "model": APIResponse},
        500: {"description": "Server error", "model": APIResponse},
    },
    summary="Generate project diagrams from proposal",
    description=(
        "Complete FlowForge workflow endpoint. Accepts a client proposal and user prompt, "
        "runs the full agent pipeline (Timeline -> Plan -> Image Generation -> Validation), "
        "and returns all generated diagrams with validation results."
    ),
)
async def generate_diagrams(
    request: DiagramGenerationRequest,
    hf_token: str = Depends(get_hf_token),
) -> WorkflowCompleteResponse:
    """
    Execute the complete FlowForge workflow to generate project diagrams.

    **Flow:**
    1. **Prompt Optimization** (LangChain) - Optimizes the user prompt for agent consumption
    2. **Timeline Agent** - Generates milestones, parallel work streams, and Gantt chart
    3. **Plan Agent** - Creates detailed project plan for diagram generation
    4. **Image Generator Agent** - Produces requested diagram types (workflow, CI/CD, etc.)
    5. **Validator Agent** - Validates all generated diagrams for correctness

    **Parameters:**
    - `proposal`: Client's project proposal with title, description, requirements
    - `prompt`: User's specific instructions and diagram type preferences
    - `hf_token`: HuggingFace API token (auto-injected from config)
    """
    try:
        proposal_data = request.proposal
        prompt_data = request.prompt

        # Build proposal text from structured input
        proposal_parts = [
            f"Project: {proposal_data.title}",
            f"\nDescription:\n{proposal_data.description}",
        ]

        if proposal_data.requirements:
            proposal_parts.append(
                "\nRequirements:\n"
                + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(proposal_data.requirements))
            )

        if proposal_data.constraints:
            proposal_parts.append(
                "\nConstraints:\n"
                + "\n".join(f"  - {c}" for c in proposal_data.constraints)
            )

        if proposal_data.tech_stack:
            proposal_parts.append(
                f"\nTechnology Stack: {', '.join(proposal_data.tech_stack)}"
            )

        if proposal_data.budget_range:
            proposal_parts.append(f"\nBudget Range: {proposal_data.budget_range}")

        proposal_text = "\n".join(proposal_parts)

        # Run the complete workflow
        result = run_flowforge_workflow(
            proposal=proposal_text,
            prompt=prompt_data.user_prompt,
            hf_token=hf_token,
            project_title=proposal_data.title,
            timeline_weeks=proposal_data.timeline_weeks,
            team_size=proposal_data.team_size,
            tech_stack=proposal_data.tech_stack or [],
            priority=prompt_data.priority,
            diagram_types=[dt.value for dt in prompt_data.diagram_types],
            optimize_prompt=prompt_data.optimize_prompt,
        )

        return format_workflow_response(result)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(exc)}",
        )
    

@router.get(
    "/health",
    response_model=APIResponse,
    summary="Health check",
    description="Verify API is running and HF_TOKEN is configured.",
)
async def health_check(hf_token: str = Depends(get_hf_token)) -> APIResponse:
    """Health check endpoint."""
    return APIResponse(
        success=True,
        message="FlowForge API is healthy",
        data={"hf_token_configured": bool(hf_token)},
    )