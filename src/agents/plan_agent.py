"""Plan Agent for FlowForge - generates detailed project plans from timelines."""

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from src.agents.base_agent import BaseAgent
from src.config import Config


class PlanAgent(BaseAgent):
    """Generate detailed project plans from timelines with resource allocation and risk analysis."""

    PLAN_SYSTEM_PROMPT = """You are an expert project manager and strategic planner
with deep experience in software engineering projects. You create comprehensive,
actionable project plans that account for resources, risks, dependencies,
and quality assurance.

Your plans must be:
- Realistic and achievable within the given timeline
- Resource-efficient with clear role assignments
- Risk-aware with mitigation strategies
- Quality-focused with explicit QA procedures
- Communication-rich with clear reporting structures

For each task, consider:
- Effort estimation in hours/days
- Required skills and team members
- Dependencies on other tasks
- Risk level and mitigation approach
- Definition of done"""

    def __init__(self) -> None:
        """Initialize the plan agent."""
        super().__init__("plan_agent")

        self.llm: HuggingFaceEndpoint | None = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """
        Initialize Hugging Face LLM.

        Raises
        ------
        ValueError
            If HF_TOKEN is missing.
        """
        if not Config.HF_TOKEN:
            raise ValueError("HF_TOKEN must be set.")

        self.llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1",
            task="text-generation",
            max_new_tokens=2048,
            temperature=0.5,
            top_p=0.9,
            huggingfacehub_api_token=Config.HF_TOKEN,
        )

    def _build_plan_prompt(
        self,
        timetable: str,
        proposal_context: str = "",
        team_size: int = 5,
        priority: str = "medium",
    ) -> str:
        """Build a structured plan prompt from timetable."""

        priority_instructions = {
            "low": "Relaxed pace with thorough documentation.",
            "medium": "Balanced pace with good coverage of all aspects.",
            "high": "Aggressive timeline with focus on MVP delivery.",
            "critical": "Maximum urgency, prioritize critical path items.",
        }.get(priority, "Balanced pace.")

        template = PromptTemplate(
            input_variables=[
                "timetable",
                "proposal_context",
                "team_size",
                "priority_instructions",
            ],
            template="""
{system_prompt}

Priority Level Context: {priority_instructions}
Team Size: {team_size} members

Timeline Reference:
{proposal_context}

Detailed Timetable:
{timetable}

Generate a comprehensive project plan with the following sections:

## 1. Task Breakdown
Break down all deliverables into atomic tasks with:
- Task name
- Owner/role
- Estimated effort (hours)
- Priority (P0/P1/P2/P3)
- Status tracking fields

## 2. Resource Allocation
- Map tasks to team roles ({team_size} members)
- Identify skill requirements per task
- Flag resource bottlenecks and overload risks
- Suggest hiring/contractor needs if gaps exist

## 3. Risk Mitigation
- Identify top 10 risks with probability and impact scores
- Define mitigation strategies for each risk
- Create contingency plans for critical risks
- Specify risk owners

## 4. Communication Structure
- Define reporting cadence (daily standups, weekly reviews)
- Specify stakeholder update frequency
- Define escalation paths
- Document decision-making authority

## 5. QA Procedures
- Define testing strategy (unit, integration, E2E, performance)
- Code review requirements
- Acceptance criteria per milestone
- Security review checkpoints

## 6. Budget Estimation
- Break down costs by phase
- Include personnel, infrastructure, tooling, and contingency
- Provide total estimate with confidence range

Be thorough and realistic in all estimates.""".format(
                system_prompt=self.PLAN_SYSTEM_PROMPT,
                timetable=timetable,
                proposal_context=proposal_context,
                team_size=team_size,
                priority_instructions=priority_instructions,
            ),
        )
        return template.format(
            timetable=timetable,
            proposal_context=proposal_context,
            team_size=team_size,
            priority_instructions=priority_instructions,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a detailed project plan.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing timetable and project context.

        Returns
        -------
        dict[str, Any]
            Updated workflow state with detailed plan.
        """
        try:
            timetable = state.get("timetable", "")
            proposal = state.get("proposal", state.get("prompt", ""))
            team_size = state.get("team_size", 5)
            priority = state.get("priority", "medium")

            if not timetable:
                return self._update_state(
                    state,
                    {
                        "error": "No timetable provided for plan agent.",
                        "plan": None,
                        "current_agent": "plan_agent",
                    },
                )

            self.logger.info("Generating detailed project plan.")

            formatted_prompt = self._build_plan_prompt(
                timetable=timetable,
                proposal_context=proposal,
                team_size=team_size,
                priority=priority,
            )

            response = self.llm.invoke(formatted_prompt)
            plan = (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )

            self.logger.info("Plan generated successfully.")

            return self._update_state(
                state,
                {
                    "plan": plan,
                    "error": None,
                    "current_agent": "plan_agent",
                },
            )

        except Exception as exc:
            error_msg = f"Plan agent failed: {exc}"
            self.logger.error(error_msg, exc_info=True)

            return self._update_state(
                state,
                {
                    "error": "Failed to generate plan",
                    "plan": None,
                    "current_agent": "plan_agent",
                },
            )