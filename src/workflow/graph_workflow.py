"""LangGraph workflow for FlowForge architecture.

Implements the agent pipeline:
  1. Prompt Optimization (LangChain)
  2. TimeAgent -> milestones, parallel work, Gantt chart
  3. PlanAgent -> detailed plan for diagrams
  4. ImageGeneratorAgent -> multiple diagrams (workflow, CI/CD, system design, etc.)
  5. ValidatorAgent -> validates all generated diagrams
"""

from typing import Any, Optional

from langgraph.graph import StateGraph, END

from src.agents.time_agent import TimeAgent
from src.agents.plan_agent import PlanAgent
from src.agents.image_generator_agent import ImageGeneratorAgent
from src.agents.validator_agent import ValidatorAgent
from src.pipeline.prompt_optimizer import PromptOptimizer


class FlowForgeState(dict):
    """State definition for the FlowForge workflow.

    Extends dict for compatibility with LangGraph TypedDict expectations
    while providing attribute-style access.
    """

    # Inputs
    proposal: str
    prompt: str
    hf_token: str

    # Project configuration
    project_title: str
    timeline_weeks: int
    team_size: int
    tech_stack: list[str]
    priority: str
    diagram_types: list[str]

    # Intermediate outputs
    optimized_prompt: Optional[str]
    timetable: Optional[str]
    milestones: list[str]
    parallel_work_streams: list[str]
    plan: Optional[str]

    # Image generation outputs
    diagrams: list[dict]
    diagram_count: int
    valid_diagram_count: int

    # Validation outputs
    validation_results: list[dict]
    overall_validation: bool
    feedback: Optional[str]

    # Control flow
    current_agent: Optional[str]
    error: Optional[str]

    def __init__(self, **kwargs):
        super().__init__()
        self.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'FlowForgeState' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def route_time(state: FlowForgeState) -> str:
    """Route after time agent: proceed to plan or end on error."""
    return "plan_agent" if not state.get("error") else "end"


def route_plan(state: FlowForgeState) -> str:
    """Route after plan agent: proceed to image generation or end on error."""
    return "image_agent" if not state.get("error") else "end"


def route_image(state: FlowForgeState) -> str:
    """Route after image agent: proceed to validation or end on error."""
    return "validator_agent" if not state.get("error") else "end"


def route_validator(state: FlowForgeState) -> str:
    """Route after validator agent: always end."""
    return "end"


def create_flowforge_workflow() -> StateGraph:
    """
    Create and configure the FlowForge LangGraph workflow.

    Returns
    -------
        StateGraph: Configured workflow graph
    """
    # Initialize agents
    time_agent = TimeAgent()
    plan_agent = PlanAgent()
    image_agent = ImageGeneratorAgent()
    validator_agent = ValidatorAgent()

    # Define the workflow
    workflow = StateGraph(FlowForgeState)

    # Add all agent nodes
    workflow.add_node("time_agent", time_agent.execute)
    workflow.add_node("plan_agent", plan_agent.execute)
    workflow.add_node("image_agent", image_agent.execute)
    workflow.add_node("validator_agent", validator_agent.execute)

    workflow.set_entry_point("time_agent")

    # Time -> Plan or END (if error)
    workflow.add_conditional_edges(
        "time_agent",
        route_time,
        {
            "plan_agent": "plan_agent",
            "end": END,
        },
    )

    # Plan -> Image or END (if error)
    workflow.add_conditional_edges(
        "plan_agent",
        route_plan,
        {
            "image_agent": "image_agent",
            "end": END,
        },
    )

    # Image -> Validator or END (if error)
    workflow.add_conditional_edges(
        "image_agent",
        route_image,
        {
            "validator_agent": "validator_agent",
            "end": END,
        },
    )

    # Validator -> END
    workflow.add_conditional_edges(
        "validator_agent",
        route_validator,
        {
            "end": END,
        },
    )

    return workflow


# Module-level optimizer instance (lazy-initialized)
_optimizer: Optional[PromptOptimizer] = None


def get_optimizer() -> PromptOptimizer:
    """Get or create the singleton PromptOptimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptOptimizer()
    return _optimizer


def run_flowforge_workflow(
    proposal: str,
    prompt: str,
    hf_token: str,
    project_title: str = "",
    timeline_weeks: int = 12,
    team_size: int = 5,
    tech_stack: list[str] | None = None,
    priority: str = "medium",
    diagram_types: list[str] | None = None,
    optimize_prompt: bool = True,
) -> FlowForgeState:
    """
    Execute the FlowForge workflow with given inputs.

    Args:
        proposal: Client project proposal text
        prompt: User's detailed prompt/instructions
        hf_token: HuggingFace token for model access
        project_title: Optional project title override
        timeline_weeks: Estimated project duration in weeks
        team_size: Number of team members
        tech_stack: Preferred technology stack
        priority: Project priority (low, medium, high, critical)
        diagram_types: List of diagram types to generate
        optimize_prompt: Whether to apply LangChain prompt optimization

    Returns:
        FlowForgeState: Final state after workflow execution
    """
    # Step 0: Prompt optimization with LangChain
    optimized_prompt = None
    working_prompt = prompt

    if optimize_prompt:
        try:
            optimizer = get_optimizer()
            optimization_result = optimizer.optimize(prompt)
            working_prompt = optimization_result["optimized_prompt"]
            optimized_prompt = working_prompt
        except Exception as exc:
            # Log but continue with original prompt if optimization fails
            import logging

            logger = logging.getLogger("app")
            logger.warning(
                "Prompt optimization failed, using original prompt: %s", exc
            )

    # Build the initial workflow state
    workflow = create_flowforge_workflow().compile()

    initial_state = FlowForgeState(
        proposal=proposal,
        prompt=working_prompt,
        hf_token=hf_token,
        project_title=project_title or proposal[:100],
        timeline_weeks=timeline_weeks,
        team_size=team_size,
        tech_stack=tech_stack or [],
        priority=priority,
        diagram_types=diagram_types or ["workflow", "ci_cd"],
        optimized_prompt=optimized_prompt,
        timetable=None,
        milestones=[],
        parallel_work_streams=[],
        plan=None,
        diagrams=[],
        diagram_count=0,
        valid_diagram_count=0,
        validation_results=[],
        overall_validation=False,
        feedback=None,
        current_agent=None,
        error=None,
    )

    # Execute the workflow
    final_state = workflow.invoke(dict(initial_state))
    final_state["current_agent"] = "validator_agent"

    # Wrap results back into FlowForgeState
    result = FlowForgeState(**final_state)

    # Post-process - summarize proposal
    if result.get("timetable"):
        result["proposal_summary"] = _summarize_proposal(
            proposal, result.get("timetable", "")
        )
    else:
        result["proposal_summary"] = proposal[:200]

    return result


def _summarize_proposal(proposal: str, timetable: str) -> str:
    """Generate a concise summary of the proposal and timeline."""
    lines = proposal.split("\n")
    summary = lines[0][:200] if lines else proposal[:200]
    if len(proposal) > 200:
        summary += "..."
    return summary