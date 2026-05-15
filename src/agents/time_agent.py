"""Time Agent for FlowForge - generates project timelines with milestones and parallel work streams."""

import time
from typing import Any, Optional

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from src.agents.base_agent import BaseAgent
from src.config import Config


class TimeAgent(BaseAgent):
    """Generate project timelines from user proposals with milestones, parallel work streams, and Gantt charts."""

    TIMELINE_SYSTEM_PROMPT = """You are a senior project planning expert. Also if the input does not have team members/team size speculate and evaluate based on the input. Output ONLY the following four sections in plain text. No markdown headings (##, ###). No prose outside the sections.

SECTION 1 - PHASES:
Phase Name: <name>
- Duration: <duration>
- Milestones: <milestones>
- Entry Criteria: <criteria>
- Exit Criteria: <criteria>

(repeat for each phase)

SECTION 2 - PARALLEL STREAMS:
- Stream: <name>: <description>

(list 2-4 streams, e.g. Backend, Frontend, DevOps, QA)

SECTION 3 - DEPENDENCIES:
- <dependency description>

SECTION 4 - GANTT CHART:
Output ONLY a valid Mermaid gantt block. No text before or after it. No markdown fences.

gantt
    title <Project Title> Timeline
    dateFormat YYYY-MM-DD

    section Discovery
    Requirement Analysis :a1, 2026-05-10, 5d

    section Design
    System Design :a2, after a1, 7d

    section Development
    Backend Development :a3, after a2, 14d
    Frontend Development :a4, after a2, 14d

    section Testing
    QA Testing :a5, after a3, 7d

    section Deployment
    Release :a6, after a5, 3d"""
    def __init__(self, session_manager: Optional[Any] = None) -> None:
        """Initialize the time agent."""
        super().__init__("time_agent", session_manager=session_manager)

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

        return f"""{self.TIMELINE_SYSTEM_PROMPT}

Project: {project_title or "Project"}
Target Duration: {timeline_weeks} weeks
Team Size: {team_size} members{tech_context}

Project Proposal:
{proposal}

Now output the four sections exactly as shown above for this project. End with the gantt block. No markdown fences around the gantt block. No ## headings."""

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
            msg = response.choices[0].message
            timetable = (
                (msg.content and msg.content.strip())
                or (getattr(msg, "reasoning_content", None) or "").strip()
                or (getattr(msg, "reasoning", None) or "").strip()
                or ""
            )

            # Extract milestones from response
            milestones = self._extract_milestones(timetable)
            parallel_streams = self._extract_parallel_streams(timetable)

            self.logger.info("Timeline generated successfully.")

            result_state = self._update_state(
                state,
                {
                    "timetable": timetable,
                    "milestones": milestones,
                    "parallel_work_streams": parallel_streams,
                    "error": None,
                    "current_agent": "time_agent",
                },
            )

            self._save_session_output(
                output_type="timeline",
                output_data={
                    "timetable": timetable,
                    "milestones": milestones,
                    "parallel_work_streams": parallel_streams,
                },
                feedback=result_state.get("error"),
                is_valid=1 if not result_state.get("error") else 0,
            )

            return result_state

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
        in_streams_section = False
        for line in timetable.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            # Detect section 2 header (handles both "SECTION 2" and "## 2." formats)
            if "parallel" in line_lower and ("stream" in line_lower or "section 2" in line_lower):
                in_streams_section = True
                continue
            # Stop at next section
            if in_streams_section and line_lower.startswith("section") and "parallel" not in line_lower:
                break
            if in_streams_section and ("section 3" in line_lower or "dependenc" in line_lower or "gantt" in line_lower):
                break
            if in_streams_section and line_stripped.startswith("-") and len(line_stripped) > 5:
                streams.append(line_stripped.lstrip("- ").strip())
        return streams