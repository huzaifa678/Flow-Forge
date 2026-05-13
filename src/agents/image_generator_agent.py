"""Image Generator Agent for FlowForge - supports multiple diagram types."""

import base64
import re
import time
from typing import Any, Optional

import requests
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.prompts import PromptTemplate

from src.agents.base_agent import BaseAgent
from src.config import Config
from src.schemas.request import DiagramType

try:
    from mermaid import Mermaid
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False
    Mermaid = None


class ImageGeneratorAgent(BaseAgent):
    """Generate Mermaid diagrams from project plans."""

    DIAGRAM_TEMPLATES = {
        DiagramType.WORKFLOW: {
            "system_role": (
                "You are an expert workflow diagram designer."
            ),
            "requirements": (
                "- Use flowchart syntax\n"
                "- Include decision nodes\n"
                "- Show process flow clearly\n"
                "- Include start and end nodes"
            ),
        },
        DiagramType.CI_CD: {
            "system_role": (
                "You are an expert CI/CD architect."
            ),
            "requirements": (
                "- Show build/test/deploy stages\n"
                "- Include rollback paths\n"
                "- Show branching strategy"
            ),
        },
        DiagramType.SYSTEM_DESIGN: {
            "system_role": (
                "You are an expert system architect."
            ),
            "requirements": (
                "- Include services/databases/caches\n"
                "- Show data flow directions\n"
                "- Include API gateways"
            ),
        },
        DiagramType.FLOWCHART: {
            "system_role": (
                "You are an expert business process analyst."
            ),
            "requirements": (
                "- Use proper flowchart syntax\n"
                "- Include conditional branches\n"
                "- Use readable structure"
            ),
        },
        DiagramType.ARCHITECTURE: {
            "system_role": (
                "You are an enterprise architect."
            ),
            "requirements": (
                "- Show architecture layers\n"
                "- Include network boundaries\n"
                "- Show infrastructure components"
            ),
        },
        DiagramType.GANTT: {
            "system_role": (
                "You are an expert project manager."
            ),
            "requirements": (
                "- Use gantt syntax only\n"
                "- Include milestones\n"
                "- Show dependencies"
            ),
        },
    }

    def __init__(self, session_manager: Optional[Any] = None) -> None:
        """Initialize image generator agent."""
        super().__init__("image_agent", session_manager=session_manager)

        self.llm: Optional[InferenceClient] = None

        self.logger.info("Initializing ImageGeneratorAgent.")

        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize Hugging Face LLM."""
        self.logger.info("Initializing image generation LLM.")

        if not Config.HF_TOKEN:
            self.logger.error("HF_TOKEN missing in configuration.")
            raise ValueError("HF_TOKEN must be set.")

        self.llm = InferenceClient(
            model="deepseek-ai/DeepSeek-R1",
            token=Config.HF_TOKEN,
        )

        self.logger.info(
            "Image generator LLM initialized successfully with model=%s",
            "deepseek-ai/DeepSeek-R1",
        )

    def _extract_text(self, response: Any) -> str:
        """Safely extract text from LLM response."""
        self.logger.info(
            "Extracting text from response type=%s",
            type(response).__name__,
        )

        if response is None:
            self.logger.warning("Received empty response object.")
            return ""

        if isinstance(response, str):
            self.logger.info(
                "Response is raw string with length=%d",
                len(response),
            )
            return response

        extracted = getattr(response, "content", str(response))

        self.logger.info(
            "Extracted response text length=%d",
            len(extracted),
        )

        return extracted

    def _build_diagram_prompt(
        self,
        plan: str,
        diagram_type: DiagramType,
        timeline: Optional[str] = None,
    ) -> str:
        """Build prompt for diagram generation."""
        self.logger.info(
            "Building prompt for diagram_type=%s",
            diagram_type.value,
        )

        self.logger.info(
            "Plan length=%d | Timeline length=%d",
            len(plan),
            len(timeline) if timeline else 0,
        )

        template_config = self.DIAGRAM_TEMPLATES.get(
            diagram_type,
            self.DIAGRAM_TEMPLATES[DiagramType.FLOWCHART],
        )

        prompt_template = PromptTemplate(
    input_variables=[
        "system_role",
        "rules",
        "plan",
        "timeline",
        "diagram_type_name",
    ],
    template="""
{system_role}

Rules:
{rules}

Plan:
{plan}

Timeline Reference:
{timeline}

Generate ONLY valid Mermaid syntax.
Do NOT include markdown fences.
Do NOT explain anything.
""",
)

        formatted_prompt = prompt_template.format(
            system_role=template_config["system_role"],
            rules=template_config["requirements"],
            plan=plan,
            timeline=timeline or "No timeline provided",
            diagram_type_name=diagram_type.value,
        )

        self.logger.info(
            "Prompt built successfully | length=%d",
            len(formatted_prompt),
        )

        self.logger.info(
            "Prompt preview:\n%s",
            formatted_prompt[:1000],
        )

        return formatted_prompt

    def _parse_diagram(self, response: str) -> str:
        """Extract Mermaid diagram from response."""
        self.logger.info("Parsing Mermaid diagram from LLM response.")

        diagram = self._extract_text(response).strip()

        self.logger.info(
            "Raw diagram length=%d",
            len(diagram),
        )

        if diagram.startswith("```"):
            self.logger.warning(
                "Markdown fences detected. Cleaning response."
            )

            diagram = (
                diagram.replace("```mermaid", "")
                .replace("```", "")
                .strip()
            )

        self.logger.info(
            "Parsed diagram length=%d",
            len(diagram),
        )

        self.logger.info(
            "Diagram preview:\n%s",
            diagram[:1000],
        )

        return diagram

    def _validate_diagram(self, diagram: str) -> bool:
        """Validate Mermaid syntax structure."""
        self.logger.info("Validating generated Mermaid diagram.")

        if not diagram or len(diagram.strip()) < 10:
            self.logger.warning(
                "Diagram too short or empty."
            )
            return False

        valid_starts = [
            "gantt",
            "flowchart",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram",
            "erDiagram",
            "graph",
        ]

        diagram_lower = diagram.lower().strip()

        self.logger.info(
            "Diagram starts with: %s",
            diagram_lower[:50],
        )

        is_valid = any(
            diagram_lower.startswith(v.lower())
            for v in valid_starts
        )

        self.logger.info(
            "Diagram validation result=%s",
            is_valid,
        )

        return is_valid

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Check whether error is retryable."""
        self.logger.warning(
            "Checking retryable status for error=%s",
            exc,
        )

        status_code = None

        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)

            if response:
                status_code = getattr(response, "status_code", None)

            if status_code is None:
                status_code = getattr(exc, "status_code", None)

            if status_code is None and exc.args:
                match = re.search(r"(\d{3})\s", str(exc.args[0]))
                if match:
                    status_code = int(match.group(1))

            self.logger.warning(
                "HF HTTP status code=%s",
                status_code,
            )

            return status_code in (
                408,
                429,
                500,
                502,
                503,
                504,
            )

        if isinstance(exc, requests.exceptions.HTTPError):
            response = getattr(exc, "response", None)
            if response:
                status_code = getattr(response, "status_code", None)
            self.logger.warning(
                "Requests HTTP status code=%s",
                status_code,
            )
            return status_code in (
                408,
                429,
                500,
                502,
                503,
                504,
            )

        retryable = isinstance(
            exc,
            (TimeoutError, ConnectionError),
        )

        self.logger.warning(
            "Retryable generic exception=%s",
            retryable,
        )

        return retryable

    def _chat_completion_with_retry(
        self,
        messages: list,
        max_tokens: int,
    ) -> Any:
        """Execute LLM request with retries."""
        last_exception = None

        max_attempts = 3
        base_delay = 1.0

        self.logger.info(
            "Starting diagram LLM request | max_tokens=%d",
            max_tokens,
        )

        self.logger.info(
            "Message preview:\n%s",
            messages[0]["content"][:1000],
        )

        for attempt in range(max_attempts):
            try:
                self.logger.info(
                    "LLM attempt %d/%d",
                    attempt + 1,
                    max_attempts,
                )

                start_time = time.time()

                response = self.llm.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                )

                elapsed = round(time.time() - start_time, 2)

                self.logger.info(
                    "LLM request succeeded in %.2fs",
                    elapsed,
                )

                if hasattr(response, "choices"):
                    content = response.choices[0].message.content

                    self.logger.info(
                        "Response content length=%d",
                        len(content) if content else 0,
                    )

                    self.logger.info(
                        "Response preview:\n%s",
                        content[:1000] if content else "EMPTY",
                    )

                return response

            except Exception as exc:
                last_exception = exc

                self.logger.error(
                    "LLM request failed on attempt %d/%d | error=%s",
                    attempt + 1,
                    max_attempts,
                    exc,
                    exc_info=True,
                )

                if not self._is_retryable_error(exc):
                    self.logger.error(
                        "Encountered non-retryable error."
                    )
                    raise exc

                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)

                    self.logger.warning(
                        "Retrying in %.1f seconds.",
                        delay,
                    )

                    time.sleep(delay)

        self.logger.error(
            "All LLM attempts failed."
        )

        raise last_exception

    def generate_diagram(
        self,
        plan: str,
        diagram_type: DiagramType,
        timeline: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a Mermaid diagram."""
        try:
            self.logger.info(
                "Generating diagram type=%s",
                diagram_type.value,
            )

            if not plan:
                self.logger.error(
                    "Plan missing. Cannot generate diagram."
                )

                return {
                    "diagram_type": diagram_type.value,
                    "mermaid_code": None,
                    "is_valid": False,
                    "error": "No plan provided",
                    "image_data": None,
                }

            self.logger.info(
                "Plan length=%d",
                len(plan),
            )

            prompt = self._build_diagram_prompt(
                plan,
                diagram_type,
                timeline,
            )

            response = self._chat_completion_with_retry(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=700,
            )

            raw_content = (
                response.choices[0].message.content
                if hasattr(response, "choices")
                else str(response)
            )

            self.logger.info(
                "Raw LLM content length=%d",
                len(raw_content) if raw_content else 0,
            )

            diagram = self._parse_diagram(raw_content)

            is_valid = self._validate_diagram(diagram)

            if not is_valid:
                self.logger.warning(
                    "Generated Mermaid diagram failed validation."
                )

                self.logger.warning(
                    "Invalid Mermaid output:\n%s",
                    diagram,
                )

                return {
                    "diagram_type": diagram_type.value,
                    "mermaid_code": diagram,
                    "is_valid": False,
                    "error": "No valid Mermaid diagram generated",
                    "image_data": None,
                }

            self.logger.info(
                "Diagram validated successfully."
            )

            image_data = None

            if MERMAID_AVAILABLE:
                max_render_attempts = 3
                render_delay = 1.0
                for render_attempt in range(max_render_attempts):
                    try:
                        self.logger.info(
                            "Attempting Mermaid PNG rendering (attempt %d/%d).",
                            render_attempt + 1,
                            max_render_attempts,
                        )

                        mm = Mermaid(diagram)

                        response = mm.img_response
                        if response.status_code == 200:
                            image_bytes = response.content

                            image_base64 = base64.b64encode(
                                image_bytes
                            ).decode("utf-8")

                            image_data = f"data:image/png;base64,{image_base64}"

                            self.logger.info(
                                "Diagram image rendered successfully."
                            )
                            break
                        else:
                            self.logger.warning(
                                "Mermaid rendering failed with status %d",
                                response.status_code,
                            )
                            if render_attempt < max_render_attempts - 1:
                                time.sleep(render_delay)
                                render_delay *= 2

                    except Exception as exc:
                        self.logger.warning(
                            "Failed Mermaid rendering | error=%s",
                            exc,
                            exc_info=True,
                        )
                        if render_attempt < max_render_attempts - 1:
                            time.sleep(render_delay)
                            render_delay *= 2

            else:
                self.logger.warning(
                    "Mermaid renderer unavailable."
                )

            return {
                "diagram_type": diagram_type.value,
                "mermaid_code": diagram,
                "title": diagram_type.value.replace("_", " ").title(),
                "description": (
                    f"Auto-generated {diagram_type.value} diagram"
                ),
                "is_valid": True,
                "validation_feedback": "Diagram generated successfully",
                "error": None,
                "image_data": image_data,
            }

        except Exception as exc:
            self.logger.error(
                "Diagram generation failed for type=%s | error=%s",
                diagram_type.value,
                exc,
                exc_info=True,
            )

            return {
                "diagram_type": diagram_type.value,
                "mermaid_code": None,
                "is_valid": False,
                "error": (
                    f"Failed to generate {diagram_type.value} "
                    f"diagram: {exc}"
                ),
                "image_data": None,
            }

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate all requested diagrams."""
        try:
            self.logger.info(
                "Starting image generation workflow."
            )

            plan = state.get("plan")
            timeline = state.get("timetable")

            improvement_prompt = state.get("improvement_prompt")
            if improvement_prompt:
                self.logger.info("Using improvement prompt for regeneration")
                plan = f"{plan}\n\n--- IMPROVEMENT FEEDBACK ---\n{improvement_prompt}"

            diagram_types_raw = state.get(
                "diagram_types",
                [DiagramType.WORKFLOW],
            )

            self.logger.info(
                "Plan exists=%s | timeline exists=%s",
                bool(plan),
                bool(timeline),
            )

            self.logger.info(
                "Requested diagram types=%s",
                diagram_types_raw,
            )

            if not plan:
                self.logger.error(
                    "No plan provided to image generator."
                )

                return self._update_state(
                    state,
                    {
                        "error": (
                            "Failed to generate images: "
                            "no plan provided"
                        ),
                        "diagrams": [],
                        "current_agent": "image_agent",
                    },
                )

            # Parse diagram types
            if isinstance(diagram_types_raw, list):
                if all(
                    isinstance(d, str)
                    for d in diagram_types_raw
                ):
                    diagram_types = [
                        DiagramType(d)
                        for d in diagram_types_raw
                    ]
                else:
                    diagram_types = diagram_types_raw

            elif isinstance(diagram_types_raw, str):
                diagram_types = [
                    DiagramType(diagram_types_raw)
                ]

            else:
                diagram_types = [DiagramType.WORKFLOW]

            self.logger.info(
                "Resolved %d diagram types.",
                len(diagram_types),
            )

            diagrams = []

            previous_invalid = self._get_previous_invalid_output("diagrams")
            if previous_invalid:
                self.logger.info(
                    "Found previous invalid output for reference"
                )

            for idx, diagram_type in enumerate(diagram_types):
                self.logger.info(
                    "Generating diagram %d/%d | type=%s",
                    idx + 1,
                    len(diagram_types),
                    diagram_type.value,
                )

                diagram_result = self.generate_diagram(
                    plan=plan,
                    diagram_type=diagram_type,
                    timeline=timeline,
                )

                self.logger.info(
                    "Diagram generation result | valid=%s | error=%s",
                    diagram_result.get("is_valid"),
                    diagram_result.get("error"),
                )

                diagrams.append(diagram_result)

            valid_count = sum(
                1 for d in diagrams if d.get("is_valid")
            )

            self.logger.info(
                "Generated %d/%d valid diagrams.",
                valid_count,
                len(diagrams),
            )

            for idx, diagram in enumerate(diagrams):
                self.logger.info(
                    "Diagram #%d summary | type=%s | valid=%s",
                    idx + 1,
                    diagram.get("diagram_type"),
                    diagram.get("is_valid"),
                )

            result_state = self._update_state(
                state,
                {
                    "diagrams": diagrams,
                    "diagram_count": len(diagrams),
                    "valid_diagram_count": valid_count,
                    "error": (
                        None
                        if valid_count > 0
                        else "No valid diagrams generated"
                    ),
                    "current_agent": "image_agent",
                },
            )

            self._save_session_output(
                output_type="diagrams",
                output_data={
                    "diagrams": diagrams,
                    "diagram_count": len(diagrams),
                    "valid_diagram_count": valid_count,
                },
                feedback=result_state.get("error"),
                is_valid=valid_count > 0,
            )

            return result_state

        except Exception as exc:
            self.logger.error(
                "Image generator workflow failed | error=%s",
                exc,
                exc_info=True,
            )

            return self._update_state(
                state,
                {
                    "error": (
                        f"Image generator agent failed: {exc}"
                    ),
                    "diagrams": [],
                    "diagram_count": 0,
                    "valid_diagram_count": 0,
                    "current_agent": "image_agent",
                },
            )