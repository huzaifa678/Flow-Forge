"""Validator Agent for FlowForge."""

import re
from typing import Any

from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_endpoint import (
    HuggingFaceEndpoint,
)

from src.agents.base_agent import BaseAgent
from src.config import Config


class ValidatorAgent(BaseAgent):
    """Validate Mermaid diagram syntax and structure."""

    def __init__(self) -> None:
        """Initialize validator agent."""
        super().__init__("validator_agent")

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
        hf_token = Config.HF_TOKEN

        if not hf_token:
            raise ValueError("HF_TOKEN must be set.")

        self.llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-VL-72B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.3,
            huggingfacehub_api_token=hf_token,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Validate Mermaid diagram.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing Mermaid diagram.

        Returns
        -------
        dict[str, Any]
            Updated workflow state.
        """
        try:
            mermaid_diagram = state.get(
                "mermaid_diagram",
                "",
            )

            if not mermaid_diagram:
                error_msg = (
                    "No Mermaid diagram provided."
                )

                self.logger.error(error_msg)

                return self._update_state(
                    state,
                    {"error": error_msg},
                )

            self.logger.info(
                "Validating Mermaid diagram."
            )

            basic_validation = (
                self._basic_mermaid_validation(
                    mermaid_diagram
                )
            )

            if not basic_validation["is_valid"]:
                return self._update_state(
                    state,
                    {
                        "validation_result": False,
                        "feedback": basic_validation[
                            "feedback"
                        ],
                        "error": None,
                    },
                )

            validation_prompt = """
            Validate the following Mermaid diagram.

            Check:
            1. Syntax
            2. Logical consistency
            3. Completeness
            4. Best practices

            Mermaid Diagram:
            {mermaid_diagram}

            Return:

            VALID: true/false
            FEEDBACK: explanation
            """

            prompt_template = PromptTemplate(
                input_variables=["mermaid_diagram"],
                template=validation_prompt,
            )

            formatted_prompt = prompt_template.format(
                mermaid_diagram=mermaid_diagram
            )

            llm_response = self.llm.invoke(
                formatted_prompt
            )

            validation_result, feedback = (
                self._parse_validation_response(
                    llm_response
                )
            )

            self.logger.info(
                "Validation completed."
            )

            return self._update_state(
                state,
                {
                    "validation_result": validation_result,
                    "feedback": feedback,
                    "error": None,
                },
            )

        except Exception as exc:
            error_msg = (
                f"Validator agent failed: {exc}"
            )

            self.logger.error(
                error_msg,
                exc_info=True,
            )

            return self._update_state(
                state,
                {"error": error_msg},
            )

    def _basic_mermaid_validation(
        self,
        diagram: str,
    ) -> dict[str, Any]:
        """
        Perform basic Mermaid validation.

        Parameters
        ----------
        diagram : str
            Mermaid diagram string.

        Returns
        -------
        dict[str, Any]
            Validation result dictionary.
        """
        if not diagram.strip():
            return {
                "is_valid": False,
                "feedback": "Empty diagram.",
            }

        mermaid_starts = [
            "gantt",
            "flowchart",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram",
            "erDiagram",
        ]

        diagram_lower = diagram.lower().strip()

        starts_with_mermaid = any(
            diagram_lower.startswith(keyword.lower())
            for keyword in mermaid_starts
        )

        if not starts_with_mermaid:
            return {
                "is_valid": False,
                "feedback": (
                    "Diagram does not start with "
                    "valid Mermaid syntax."
                ),
            }

        if diagram.count("[") != diagram.count("]"):
            return {
                "is_valid": False,
                "feedback": "Mismatched brackets.",
            }

        if diagram.count("(") != diagram.count(")"):
            return {
                "is_valid": False,
                "feedback": "Mismatched parentheses.",
            }

        if diagram.count("{") != diagram.count("}"):
            return {
                "is_valid": False,
                "feedback": "Mismatched braces.",
            }

        return {
            "is_valid": True,
            "feedback": "Validation passed.",
        }

    def _parse_validation_response(
        self,
        response: str,
    ) -> tuple[bool, str]:
        """
        Parse validation response.

        Parameters
        ----------
        response : str
            Raw LLM response.

        Returns
        -------
        tuple[bool, str]
            Validation result and feedback.
        """
        response = response.strip()

        valid_match = re.search(
            r"VALID:\s*(true|false)",
            response,
            re.IGNORECASE,
        )

        if valid_match:
            validation_result = (
                valid_match.group(1).lower() == "true"
            )
        else:
            validation_result = False

        feedback_match = re.search(
            r"FEEDBACK:\s*(.*)",
            response,
            re.IGNORECASE | re.DOTALL,
        )

        if feedback_match:
            feedback = feedback_match.group(1).strip()
        else:
            feedback = "No feedback provided."

        return validation_result, feedback
    