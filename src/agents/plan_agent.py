"""Plan Agent for FlowForge - generates detailed project plans from timelines."""

import time
from typing import Any

import concurrent.futures
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.prompts import PromptTemplate

from src.agents.base_agent import BaseAgent
from src.config import Config, LLMConfig


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

        self.llm: InferenceClient | None = None
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

        self.llm = InferenceClient(
            model="deepseek-ai/DeepSeek-R1",
            token=Config.HF_TOKEN,
            provider=LLMConfig.PLAN_PROVIDER,
        )

    def _build_plan_prompt(
        self,
        timetable: str,
        proposal_context: str = "",
        team_size: int = 5,
        priority: str = "medium",
    ) -> str:
        """Build a structured plan prompt from timetable."""

        self.logger.info("Building plan prompt with team_size=%d and priority=%s", team_size, priority)

        priority_instructions = {
            "low": "Relaxed pace with thorough documentation.",
            "medium": "Balanced pace with good coverage of all aspects.",
            "high": "Aggressive timeline with focus on MVP delivery.",
            "critical": "Maximum urgency, prioritize critical path items.",
        }.get(priority, "Balanced pace.")

        self.logger.info("Priority instructions set: %s", priority_instructions)

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

        self.logger.info("Plan prompt template created successfully.")

        formatted_prompt = template.format(
            timetable=timetable,
            proposal_context=proposal_context,
            team_size=team_size,
            priority_instructions=priority_instructions,
        )

        self.logger.info("Plan prompt formatted successfully with length %d characters.", len(formatted_prompt))

        return formatted_prompt

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Check if an exception is retryable (timeout, server errors, etc.)."""
        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)
            if response is not None:
                status_code = getattr(response, "status_code", None)
                if status_code is not None:
                    # Retry on 408 (timeout), 429 (rate limit), and any 5xx server error
                    return status_code in (408, 429) or (500 <= status_code < 600)
            # Fallback: check error message for known patterns
            error_msg = str(exc).lower()
            if any(
                code in error_msg
                for code in ("504", "502", "503", "500", "gateway timeout", "service unavailable")
            ):
                return True
            # Default to retryable for unknown HfHubHTTPError (safer for transient issues)
            return True
        return isinstance(exc, (TimeoutError, ConnectionError))
    
    def _call_with_timeout(self, func, timeout: int, *args, **kwargs):
        """
        Call a function with a forced timeout using concurrent.futures.
        Args:            func: The function to call.
            timeout: Maximum time in seconds to allow for the function to execute.
            *args, **kwargs: Arguments to pass to the function.
        Returns:            The result of the function call if it completes within the timeout.
        Raises:            TimeoutError: If the function call exceeds the specified timeout.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            return future.result(timeout=timeout)

    def _chat_completion_with_retry(
        self, messages: list, max_tokens: int, formatted_prompt: str) -> Any:
        """Execute chat completion with retry logic and forced timout for transient failures."""
        last_exception = None
        max_attempts = 2
        base_delay = 1.0
        request_timeout = getattr(self, "request_timeout", 180)

        self.logger.info(
            "LLM request starting | tokens=%s | prompt_preview=%s",
            max_tokens,
            formatted_prompt[:10],
        )

        for attempt in range(max_attempts):
            try:
                response = self._call_with_timeout(
                    self.llm.chat_completion,
                    request_timeout,
                    messages=messages,
                    max_tokens=max_tokens,
                )

                self.logger.info("LLM request successful on attempt %d/%d", attempt + 1, max_attempts)

                return response
            except Exception as exc:
                last_exception = exc
                if not self._is_retryable_error(exc):
                    raise exc
                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)
                    self.logger.warning(
                        "LLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_attempts,
                        delay,
                        exc,
                        formatted_prompt[:500],
                    )
                    time.sleep(delay)

        raise last_exception

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

            self.logger.info("Plan agent prompt: %s", formatted_prompt[:10])

            response = self._chat_completion_with_retry(
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt,
                    }
                ],
                max_tokens=1700,
                formatted_prompt=formatted_prompt,
            )

            plan = ""
            message = response.choices[0].message
            
            if hasattr(message, "content") and message.content:
                plan = message.content.strip()
            elif hasattr(message, "reasoning_content") and message.reasoning_content:
                plan = message.reasoning_content.strip()
            else:
                plan = str(response).strip()

            if not plan or plan == "None":
                raise ValueError("LLM returned an empty plan.")

            self.logger.info("Plan generated successfully. Length: %d", len(plan))

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