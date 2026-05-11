"""Validator Agent for FlowForge."""

import re
from typing import Any

from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

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
        """Initialize Hugging Face LLM."""
        if not Config.HF_TOKEN:
            raise ValueError("HF_TOKEN must be set.")

        self.llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-VL-72B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.3,
            huggingfacehub_api_token=Config.HF_TOKEN,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate Mermaid diagram."""
        try:
            mermaid_diagram = state.get("mermaid_diagram", "")

            if not mermaid_diagram:
                return self._update_state(
                    state,
                    {
                        "error": "No Mermaid diagram provided.",
                        "validation_result": False,
                        "feedback": None,
                        "current_agent": "validator_agent",
                    },
                )

            self.logger.info("Validating Mermaid diagram.")

            # Basic validation first
            basic_validation = self._basic_mermaid_validation(mermaid_diagram)

            if not basic_validation["is_valid"]:
                return self._update_state(
                    state,
                    {
                        "validation_result": False,
                        "feedback": basic_validation["feedback"],
                        "error": None,
                        "current_agent": "validator_agent",
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

            prompt = PromptTemplate(
                input_variables=["mermaid_diagram"],
                template=validation_prompt,
            ).format(mermaid_diagram=mermaid_diagram)

            llm_response = self.llm.invoke(prompt)

            validation_result, feedback = self._parse_validation_response(llm_response)

            return self._update_state(
                state,
                {
                    "validation_result": validation_result,
                    "feedback": feedback,
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
                    "validation_result": False,
                    "feedback": None,
                    "current_agent": "validator_agent",  
                },
            )

    def _basic_mermaid_validation(self, diagram: str) -> dict[str, Any]:
        """Basic structural validation."""
        if not diagram.strip():
            return {"is_valid": False, "feedback": "Empty diagram."}

        valid_starts = [
            "gantt",
            "flowchart",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram",
            "erDiagram",
        ]

        diagram_lower = diagram.lower().strip()

        if not any(diagram_lower.startswith(v.lower()) for v in valid_starts):
            return {
                "is_valid": False,
                "feedback": "Diagram does not start with valid Mermaid syntax.",
            }

        if diagram.count("[") != diagram.count("]"):
            return {"is_valid": False, "feedback": "Mismatched brackets."}

        if diagram.count("(") != diagram.count(")"):
            return {"is_valid": False, "feedback": "Mismatched parentheses."}

        if diagram.count("{") != diagram.count("}"):
            return {"is_valid": False, "feedback": "Mismatched braces."}

        return {"is_valid": True, "feedback": "Validation passed."}

    def _parse_validation_response(self, response: str) -> tuple[bool, str]:
        """Parse LLM validation response."""
        response = str(response).strip()

        valid_match = re.search(r"VALID:\s*(true|false)", response, re.I)
        validation_result = valid_match.group(1).lower() == "true" if valid_match else False

        feedback_match = re.search(r"FEEDBACK:\s*(.*)", response, re.I | re.S)
        feedback = feedback_match.group(1).strip() if feedback_match else "No feedback provided."

        return validation_result, feedback
    