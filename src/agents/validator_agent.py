"""Validator Agent for FlowForge - validates Mermaid diagrams with refined checks."""

import re
import time
from typing import Any

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from src.agents.base_agent import BaseAgent
from src.config import Config


class ValidatorAgent(BaseAgent):
    """Validate Mermaid diagrams for syntax correctness, logical consistency, and best practices."""

    VALIDATOR_SYSTEM_PROMPT = """You are an expert Mermaid diagram validator and code reviewer.
Your role is to thoroughly validate Mermaid.js diagrams for:

1. Syntax correctness (valid Mermaid grammar)
2. Logical consistency (no circular dependencies, proper node references)
3. Structural completeness (all nodes defined, proper connections)
4. Best practices (naming conventions, readability, organization)
5. Production readiness (error handling, scalability considerations)

Be precise in your feedback. Provide specific line numbers and concrete suggestions
for improvement. Distinguish between critical errors and style recommendations."""

    def __init__(self) -> None:
        """Initialize validator agent."""
        super().__init__("validator_agent")

        self.llm: InferenceClient | None = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize Hugging Face LLM."""
        if not Config.HF_TOKEN:
            raise ValueError("HF_TOKEN must be set.")

        self.llm = InferenceClient(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            token=Config.HF_TOKEN,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Validate all Mermaid diagrams in the workflow state.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing diagrams to validate.

        Returns
        -------
        dict[str, Any]
            Updated workflow state with validation results per diagram.
        """
        try:
            diagrams = state.get("diagrams", [])

            # Support legacy single-diagram state
            legacy_diagram = state.get("mermaid_diagram")
            if legacy_diagram and not diagrams:
                diagrams = [
                    {
                        "mermaid_code": legacy_diagram,
                        "diagram_type": "workflow",
                    }
                ]

            if not diagrams:
                return self._update_state(
                    state,
                    {
                        "error": "No diagrams to validate.",
                        "validation_results": [],
                        "overall_validation": False,
                        "current_agent": "validator_agent",
                    },
                )

            self.logger.info("Validating %d diagrams.", len(diagrams))

            validation_results = []
            all_valid = True

            for idx, diagram_data in enumerate(diagrams):
                mermaid_code = diagram_data.get("mermaid_code", "")
                diagram_type = diagram_data.get("diagram_type", "unknown")

                result = self._validate_single_diagram(mermaid_code, diagram_type)
                result["index"] = idx
                result["diagram_type"] = diagram_type
                validation_results.append(result)

                if not result["is_valid"]:
                    all_valid = False

            self.logger.info(
                "Validation complete: %d/%d diagrams valid.",
                sum(1 for r in validation_results if r["is_valid"]),
                len(validation_results),
            )

            return self._update_state(
                state,
                {
                    "validation_results": validation_results,
                    "overall_validation": all_valid,
                    "valid_count": sum(
                        1 for r in validation_results if r["is_valid"]
                    ),
                    "total_count": len(validation_results),
                    "error": None,
                    "current_agent": "validator_agent",
                },
            )

        except Exception as exc:
            error_msg = f"Validator agent failed: {exc}"
            self.logger.error(error_msg, exc_info=True)

            return self._update_state(
                state,
                {
                    "error": error_msg,
                    "validation_results": [],
                    "overall_validation": False,
                    "current_agent": "validator_agent",
                },
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

    def _validate_single_diagram(
        self, diagram: str, diagram_type: str
    ) -> dict[str, Any]:
        """
        Validate a single Mermaid diagram.

        Args:
            diagram: The Mermaid diagram code.
            diagram_type: The type of diagram being validated.

        Returns:
            Dict with validation results.
        """
        # Basic structural validation first
        basic_result = self._basic_mermaid_validation(diagram)
        if not basic_result["is_valid"]:
            return {
                **basic_result,
                "is_valid": False,
                "feedback": basic_result["feedback"],
                "suggestions": [],
            }

        # Build validation prompt using concatenation to avoid .format() issues
        # with curly braces in the template
        validation_prompt = (
            self.VALIDATOR_SYSTEM_PROMPT
            + "\n\n"
            + "Diagram Type: "
            + diagram_type
            + "\n\n"
            + "Mermaid Diagram to Validate:\n```mermaid\n"
            + diagram
            + "\n```\n\n"
            + "Perform thorough validation and return results in EXACT format:\n\n"
            + "VALID: true/false\n"
            + "CRITICAL_ISSUES: [list or \"None\"]\n"
            + "WARNINGS: [list or \"None\"]\n"
            + "SUGGESTIONS: [list or \"None\"]\n"
            + "FEEDBACK: [detailed explanation of findings]\n"
        )

        try:
            llm_response = self._chat_completion_with_retry(
                messages=[{"role": "user", "content": validation_prompt}],
                max_tokens=300,
            )

            response_text = (
                llm_response.choices[0].message.content.strip()
                if hasattr(llm_response, "choices")
                else str(llm_response).strip()
            )

            return self._parse_llm_validation(response_text)

        except Exception as exc:
            return {
                "is_valid": False,
                "feedback": f"Validation LLM call failed: {exc}",
                "critical_issues": [],
                "warnings": [],
                "suggestions": [],
            }

    def _basic_mermaid_validation(self, diagram: str) -> dict[str, Any]:
        """Basic structural validation of Mermaid syntax."""
        if not diagram or not diagram.strip():
            return {"is_valid": False, "feedback": "Empty diagram."}

        valid_starts = [
            "gantt",
            "flowchart",
            "graph",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram",
            "erDiagram",
            "pie",
            "journey",
            "gitGraph",
        ]

        diagram_lower = diagram.lower().strip()

        if not any(diagram_lower.startswith(v.lower()) for v in valid_starts):
            return {
                "is_valid": False,
                "feedback": (
                    f"Diagram does not start with valid Mermaid syntax. "
                    f"Got: '{diagram_lower[:50]}...'"
                ),
            }

        # Check bracket balance
        for open_char, close_char in [("(", ")"), ("[", "]"), ("{", "}")]:
            if diagram.count(open_char) != diagram.count(close_char):
                return {
                    "is_valid": False,
                    "feedback": f"Mismatched '{open_char}'/'{close_char}' brackets.",
                }

        return {"is_valid": True, "feedback": "Validation passed."}

    def _parse_llm_validation(self, response: str) -> dict[str, Any]:
        """Parse structured validation response from LLM."""
        result = {
            "is_valid": True,
            "feedback": "",
            "critical_issues": [],
            "warnings": [],
            "suggestions": [],
        }

        # Parse VALID field
        valid_match = re.search(r"VALID:\s*(true|false)", response, re.I)
        if valid_match:
            result["is_valid"] = valid_match.group(1).lower() == "true"

        # Parse CRITICAL_ISSUES
        critical_match = re.search(
            r"CRITICAL_ISSUES:\s*(.+?)(?:\n(?:WARNINGS|SUGGESTIONS|FEEDBACK|$))",
            response,
            re.I | re.DOTALL,
        )
        if critical_match:
            issues = critical_match.group(1).strip()
            if issues.lower() != "none":
                result["critical_issues"] = [
                    i.strip()
                    for i in re.split(r"[\n,]", issues)
                    if i.strip() and i.strip() != "-"
                ]

        # Parse WARNINGS
        warnings_match = re.search(
            r"WARNINGS:\s*(.+?)(?:\n(?:SUGGESTIONS|FEEDBACK|$))",
            response,
            re.I | re.DOTALL,
        )
        if warnings_match:
            warnings = warnings_match.group(1).strip()
            if warnings.lower() != "none":
                result["warnings"] = [
                    i.strip()
                    for i in re.split(r"[\n,]", warnings)
                    if i.strip() and i.strip() != "-"
                ]

        # Parse SUGGESTIONS
        suggestions_match = re.search(
            r"SUGGESTIONS:\s*(.+?)(?:\n(?:FEEDBACK|$))",
            response,
            re.I | re.DOTALL,
        )
        if suggestions_match:
            suggestions = suggestions_match.group(1).strip()
            if suggestions.lower() != "none":
                result["suggestions"] = [
                    i.strip()
                    for i in re.split(r"[\n,]", suggestions)
                    if i.strip() and i.strip() != "-"
                ]

        # Parse FEEDBACK
        feedback_match = re.search(r"FEEDBACK:\s*(.*)", response, re.I | re.S)
        if feedback_match:
            result["feedback"] = feedback_match.group(1).strip()

        return result