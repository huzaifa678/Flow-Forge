"""LangGraph workflow for FlowForge architecture. Implements the agent pipeline: TimeAgent -> PlanAgent -> ImageGeneratorAgent -> ValidatorAgent."""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from src.agents.time_agent import TimeAgent
from src.agents.plan_agent import PlanAgent
from src.agents.image_generator_agent import ImageGeneratorAgent
from src.agents.validator_agent import ValidatorAgent


class FlowForgeState(TypedDict):
    """State definition for the FlowForge workflow."""
    # Inputs
    prompt: str
    hf_token: str
    
    # Intermediate outputs
    timetable: Optional[str]
    plan: Optional[str]
    mermaid_diagram: Optional[str]
    
    # Final outputs
    validation_result: Optional[bool]
    feedback: Optional[str]
    
    # Control flow
    current_agent: Optional[str]
    error: Optional[str]


def create_flowforge_workflow() -> StateGraph:
    """
    Create and configure the FlowForge LangGraph workflow.
    
    Returns:
        StateGraph: Configured workflow graph
    
    """
    # Initialize agents
    time_agent = TimeAgent()
    plan_agent = PlanAgent()
    image_agent = ImageGeneratorAgent()
    validator_agent = ValidatorAgent()
    
    # Define the workflow
    workflow = StateGraph(FlowForgeState)
    
    # Add nodes (agents)
    workflow.add_node("time_agent", time_agent.execute)
    workflow.add_node("plan_agent", plan_agent.execute)
    workflow.add_node("image_agent", image_agent.execute)
    workflow.add_node("validator_agent", validator_agent.execute)
    
    # Define the flow: TimeAgent -> PlanAgent -> ImageAgent -> ValidatorAgent -> END
    workflow.set_entry_point("time_agent")
    workflow.add_edge("time_agent", "plan_agent")
    workflow.add_edge("plan_agent", "image_agent")
    workflow.add_edge("image_agent", "validator_agent")
    workflow.add_edge("validator_agent", END)
    
    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "time_agent",
        lambda state: "error" if state.get("error") else "plan_agent",
        {
            "plan_agent": "plan_agent",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "plan_agent",
        lambda state: "error" if state.get("error") else "image_agent",
        {
            "image_agent": "image_agent",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "image_agent",
        lambda state: "error" if state.get("error") else "validator_agent",
        {
            "validator_agent": "validator_agent",
            "error": END
        }
    )
    
    workflow.add_conditional_edges(
        "validator_agent",
        lambda state: "error" if state.get("error") else "end",
        {
            "end": END,
            "error": END
        }
    )
    
    return workflow.compile()


def run_flowforge_workflow(prompt: str, hf_token: str) -> FlowForgeState:
    """
    Execute the FlowForge workflow with given prompt and HF token.
    
    Args:
        prompt: User proposal/prompt for the time agent
        hf_token: HuggingFace token for model access
        
    Returns:
        FlowForgeState: Final state after workflow execution
    
    """
    workflow = create_flowforge_workflow()
    
    initial_state: FlowForgeState = {
        "prompt": prompt,
        "hf_token": hf_token,
        "timetable": None,
        "plan": None,
        "mermaid_diagram": None,
        "validation_result": None,
        "feedback": None,
        "current_agent": None,
        "error": None
    }
    
    # Execute the workflow
    final_state = workflow.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    # Example usage (for testing)
    import os
    from src.config import Config
    
    example_prompt = "Create a project plan for building a web application with user authentication"
    hf_token = os.getenv("HF_TOKEN") or Config.HF_TOKEN
    
    if not hf_token:
        raise ValueError("HF_TOKEN must be set in environment or .env file")
    
    result = run_flowforge_workflow(example_prompt, hf_token)
    print("Workflow completed:")
    print(f"Timetable: {result.get('timetable')}")
    print(f"Plan: {result.get('plan')}")
    print(f"Mermaid Diagram: {result.get('mermaid_diagram')}")
    print(f"Validation Result: {result.get('validation_result')}")
    print(f"Feedback: {result.get('feedback')}")
    