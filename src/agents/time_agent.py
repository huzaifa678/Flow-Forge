"""Time Agent for FlowForge - generates project timelines with milestones and parallel work streams."""

import time
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.prompts import PromptTemplate

from src.agents.base_agent import BaseAgent
from src.config import Config


class TimeAgent(BaseAgent):
    """Generate project timelines from user proposals with milestones, parallel work streams, and Gantt charts."""

    TIMELINE_SYSTEM_PROMPT = """You are a senior project planning expert with 20+ years of experience
managing complex software and engineering projects. You excel at creating
realistic, actionable project timelines that account for dependencies,
resource constraints, and parallel work streams.

Your task is to analyze the user's project proposal and generate:
1. A phased project timeline with clear milestones
2. Identification of parallel work streams
3. Task dependencies and critical path
4. A detailed Gantt chart representation in Mermaid syntax

Guidelines for timeline generation:
- Break the project into logical phases (Discovery, Design, Development, Testing, Deployment)
- Each phase should have clear entry and exit criteria
- Identify tasks that can run in parallel vs sequential dependencies
- Estimate realistic durations based on project complexity
- Include buffer time for reviews and iterations
- Generate a valid Mermaid gantt chart syntax
- Consider the tech stack and team size when estimating durations
- Flag high-risk dependencies and potential blockers early

Return structured output with clear sections for each component."""

    def __init__(self) -> None:
        """Initialize the time agent."""
        super().__init__("time_agent")

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
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=Config.HF_TOKEN,
        )

    def _build_timeline_prompt(
        self,
        proposal: str,
        project_title: str = "",
        timeline_weeks: int = 12,
        team_size: int = 5,
        tech_stack: list | None = None,
        include_gantt: bool = True,
        include_parallel_work: bool = True,
    ) -> str:
        """Build a structured timeline prompt from proposal context."""

        tech_context = ""
        if tech_stack:
            tech_context = f"\nTechnology Stack: {', '.join(tech_stack)}"

        timeline_instructions = "Include a Gantt chart diagram." if include_gantt else ""
        parallel_instructions = (
            "Explicitly identify parallel work streams that can run concurrently."
            if include_parallel_work
            else ""
        )

        template = PromptTemplate(
            input_variables=[
                "project_title",
                "timeline_weeks",
                "team_size",
                "tech_context",
                "timeline_instructions",
                "parallel_instructions",
                "proposal",
            ],
            template="""
{system_prompt}

Project: {project_title}
Target Duration: {timeline_weeks} weeks
Team Size: {team_size} members{tech_context}

Instructions:
{timeline_instructions}
{parallel_instructions}

Project Proposal:
{proposal}

Deliver:
## 1. Project Phases & Milestones
[List phases with milestone markers and estimated durations]

## 2. Parallel Work Streams
[Identify tasks that can run concurrently]

## 3. Dependencies & Critical Path
[List key dependencies and identify the critical path]

## 4. Gantt Chart
```mermaid
gantt
    title {project_title} - Project Timeline
    dateFormat  YYYY-MM-DD
    section Phase Name
    Task Name :a1, 2026-05-10, 10d
```
""".format(
                system_prompt=self.TIMELINE_SYSTEM_PROMPT,
                project_title=project_title or "Project",
                timeline_weeks=timeline_weeks,
                team_size=team_size,
                tech_context=tech_context,
                timeline_instructions=timeline_instructions,
                parallel_instructions=parallel_instructions,
                proposal=proposal,
            ),
        )
        return template.format(
            project_title=project_title or "Project",
            timeline_weeks=timeline_weeks,
            team_size=team_size,
            tech_context=tech_context,
            timeline_instructions=timeline_instructions,
            parallel_instructions=parallel_instructions,
            proposal=proposal,
        )

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Check if an exception is retryable (timeout, server errors, etc.)."""
        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)
            if response:
                status_code = getattr(response, "status_code", None)
                return status_code in (408, 429, 500, 502, 503, 504)
        return isinstance(exc, (TimeoutError, ConnectionError))

    def _chat_completion_with_retry(
        self, messages: list, max_tokens: int
    ) -> Any:
        """Execute chat completion with retry logic for transient failures."""
        last_exception = None
        max_attempts = 3
        base_delay = 1.0

        for attempt in range(max_attempts):
            try:
                response = self.llm.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                )
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
                    )
                    time.sleep(delay)

        raise last_exception

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate timetable and Gantt chart from proposal.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing proposal and project configuration.

        Returns
        -------
        dict[str, Any]
            Updated workflow state with timetable, milestones, and diagram.
        """
        try:
            # Support both legacy 'prompt' field and new structured 'proposal'
            proposal = state.get("proposal", state.get("prompt", ""))

            if not proposal or not proposal.strip():
                error_msg = "No proposal provided for time agent."
                self.logger.error(error_msg)
                return self._update_state(
                    state,
                    {"error": error_msg},
                )

            # Extract project configuration from state
            project_title = state.get("project_title", "")
            timeline_weeks = state.get("timeline_weeks", 12)
            team_size = state.get("team_size", 5)
            tech_stack = state.get("tech_stack", [])
            include_gantt = state.get("include_gantt", True)
            include_parallel_work = state.get("include_parallel_work", True)

            self.logger.info(
                "Generating timeline for project: %s", project_title or "(untitled)"
            )

            formatted_prompt = self._build_timeline_prompt(
                proposal=proposal,
                project_title=project_title,
                timeline_weeks=timeline_weeks,
                team_size=team_size,
                tech_stack=tech_stack,
                include_gantt=include_gantt,
                include_parallel_work=include_parallel_work,
            )

            response = self._chat_completion_with_retry(
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt,
                    }
                ],
                max_tokens=600,
            )
            timetable = (
                response.choices[0].message.content.strip()
                if hasattr(response, "choices")
                else str(response).strip()
            )

            # Extract milestones from response
            milestones = self._extract_milestones(timetable)
            parallel_streams = self._extract_parallel_streams(timetable)

            self.logger.info("Timeline generated successfully.")

            return self._update_state(
                state,
                {
                    "timetable": timetable,
                    "milestones": milestones,
                    "parallel_work_streams": parallel_streams,
                    "error": None,
                    "current_agent": "time_agent",
                },
            )

        except Exception as exc:
            error_msg = f"Time agent failed: {exc}"
            self.logger.error(error_msg, exc_info=True)

            return self._update_state(
                state,
                {
                    "error": "Failed to generate timetable",
                    "timetable": None,
                    "milestones": [],
                    "parallel_work_streams": [],
                    "current_agent": "time_agent",
                },
            )

    def _extract_milestones(self, timetable: str) -> list[str]:
        """Extract milestone markers from the timetable text."""
        milestones = []
        lines = timetable.split("\n")
        for line in lines:
            line_stripped = line.strip()
            if any(keyword in line_stripped.lower() for keyword in ["milestone", "phase", "deliverable", "checkpoint", "->"]):
                cleaned = line_stripped.lstrip("-* 0123456789.")
                if cleaned and len(cleaned) > 3:
                    milestones.append(cleaned)
        return milestones if milestones else [line.strip().lstrip("-* ") for line in lines if line.strip() and len(line.strip()) > 5][:10]

    def _extract_parallel_streams(self, timetable: str) -> list[str]:
        """Extract parallel work streams from the timetable text."""
        streams = []
        current_section = None
        for line in timetable.split("\n"):
            line_lower = line.strip().lower()
            if "section" in line_lower and "gantt" not in line_lower:
                current_section = line.strip()
            elif current_section and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                streams.append(f"{current_section}: {line.strip()}")
        return streams