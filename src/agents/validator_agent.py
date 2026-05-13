"""Validator Agent for FlowForge - validates Mermaid diagrams with refined checks."""

from typing import Any, Optional

from src.agents.base_validator_agent import BaseValidatorAgent


class ValidatorAgent(BaseValidatorAgent):
    """Validate Mermaid diagrams for syntax correctness and best practices.

    When validation fails, sets 'route_to' = 'image_agent' in the state
    so the workflow re-routes to the image generator with improvement feedback.
    """

    def __init__(self, session_manager: Optional[Any] = None) -> None:
        """Initialize validator agent."""
        super().__init__("validator_agent", session_manager=session_manager)

    def _get_model(self) -> str:
        """Return the HuggingFace model identifier."""
        return "Qwen/Qwen2.5-VL-7B-Instruct"

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Validate all Mermaid diagrams in workflow state.

        When diagrams fail validation, includes improvement feedback
        and signals the workflow to route back to the image generator.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing diagrams.

        Returns
        -------
        dict[str, Any]
            Updated workflow state with validation results and routing info.
        """
        try:
            self.logger.info("Starting diagram validation workflow.")

            diagrams = state.get("diagrams", [])

            self.logger.info(
                "Initial diagrams count from state: %d",
                len(diagrams),
            )

            # Legacy fallback
            legacy_diagram = state.get("mermaid_diagram")

            if legacy_diagram and not diagrams:
                self.logger.info(
                    "Legacy single diagram detected. Converting to diagrams list."
                )

                diagrams = [
                    {
                        "mermaid_code": legacy_diagram,
                        "diagram_type": "workflow",
                    }
                ]

            # Track previous generation prompts for re-generation
            previous_prompts = state.get("previous_prompts", [])

            if not diagrams:
                self.logger.warning("No diagrams found for validation.")

                return self._update_state(
                    state,
                    {
                        "error": "No diagrams to validate.",
                        "validation_results": [],
                        "overall_validation": False,
                        "route_to": "end",
                        "current_agent": "validator_agent",
                    },
                )

            self.logger.info(
                "Validating %d diagrams.",
                len(diagrams),
            )

            validation_results = []
            all_valid = True
            invalid_diagrams = []

            for idx, diagram_data in enumerate(diagrams):
                self.logger.info(
                    "Processing diagram %d/%d",
                    idx + 1,
                    len(diagrams),
                )

                mermaid_code = diagram_data.get("mermaid_code", "")
                diagram_type = diagram_data.get("diagram_type", "unknown")

                self.logger.info(
                    "Diagram type=%s | code_length=%d",
                    diagram_type,
                    len(mermaid_code),
                )

                self.logger.info(
                    "Diagram preview:\n%s",
                    mermaid_code[:500],
                )

                result = self._validate_single_diagram(
                    mermaid_code,
                    diagram_type,
                )

                result["index"] = idx
                result["diagram_type"] = diagram_type

                validation_results.append(result)

                self.logger.info(
                    "Validation result for %s | valid=%s",
                    diagram_type,
                    result.get("is_valid"),
                )

                self.logger.info(
                    "Validation feedback: %s",
                    result.get("feedback"),
                )

                if result.get("critical_issues"):
                    self.logger.warning(
                        "Critical issues found: %s",
                        result["critical_issues"],
                    )

                if result.get("warnings"):
                    self.logger.warning(
                        "Warnings found: %s",
                        result["warnings"],
                    )

                if not result["is_valid"]:
                    all_valid = False
                    invalid_diagrams.append({
                        "index": idx,
                        "diagram_type": diagram_type,
                        "feedback": result.get("feedback", ""),
                        "critical_issues": result.get("critical_issues", []),
                        "warnings": result.get("warnings", []),
                        "suggestions": result.get("suggestions", []),
                    })

            valid_count = sum(
                1 for r in validation_results if r["is_valid"]
            )

            self.logger.info(
                "Validation complete | valid=%d/%d",
                valid_count,
                len(validation_results),
            )

            if all_valid:
                result_state = self._update_state(
                    state,
                    {
                        "validation_results": validation_results,
                        "overall_validation": True,
                        "valid_count": valid_count,
                        "total_count": len(validation_results),
                        "error": None,
                        "route_to": "end",
                        "current_agent": "validator_agent",
                    },
                )
            else:
                # Build improvement feedback for the image generator
                improvement_prompt = self._build_improvement_prompt(
                    invalid_diagrams, previous_prompts
                )

                result_state = self._update_state(
                    state,
                    {
                        "validation_results": validation_results,
                        "overall_validation": False,
                        "valid_count": valid_count,
                        "total_count": len(validation_results),
                        "error": (
                            f"{len(invalid_diagrams)}/{len(validation_results)} "
                            f"diagrams failed validation"
                        ),
                        "improvement_prompt": improvement_prompt,
                        "failed_diagrams": invalid_diagrams,
                        "route_to": "image_agent",
                        "current_agent": "validator_agent",
                    },
                )

            self._save_session_output(
                output_type="validation_results",
                output_data={
                    "validation_results": validation_results,
                    "overall_validation": result_state.get("overall_validation"),
                    "valid_count": valid_count,
                    "total_count": len(validation_results),
                },
                feedback=result_state.get("error"),
                is_valid=result_state.get("overall_validation", False),
            )

            return result_state

        except Exception as exc:
            error_msg = f"Validator agent failed: {exc}"

            self.logger.error(
                error_msg,
                exc_info=True,
            )

            return self._update_state(
                state,
                {
                    "error": error_msg,
                    "validation_results": [],
                    "overall_validation": False,
                    "route_to": "end",
                    "current_agent": "validator_agent",
                },
            )

    def _build_improvement_prompt(
        self,
        invalid_diagrams: list[dict],
        previous_prompts: list[str],
    ) -> str:
        """Build a prompt telling the image generator what to fix.

        Parameters
        ----------
        invalid_diagrams : list[dict]
            List of failed diagram info dicts with feedback and issues.
        previous_prompts : list[str]
            Previous generation prompts to avoid repeating mistakes.

        Returns
        -------
        str
            Improvement prompt for the image generator.
        """
        prompt_parts = [
            "IMPORTANT: The previous diagram generation had validation failures. "
            "Please regenerate the diagrams with strict adherence to the following fixes:"
        ]

        for idx, diag in enumerate(invalid_diagrams):
            diag_type = diag.get("diagram_type", "unknown")
            prompt_parts.append(f"\n--- Diagram #{idx + 1} ({diag_type}) ---")

            feedback = diag.get("feedback", "")
            if feedback:
                prompt_parts.append(f"Issue: {feedback}")

            critical = diag.get("critical_issues", [])
            if critical:
                prompt_parts.append(
                    f"Critical issues: {'; '.join(critical)}"
                )

            warnings = diag.get("warnings", [])
            if warnings:
                prompt_parts.append(
                    f"Warnings: {'; '.join(warnings)}"
                )

            suggestions = diag.get("suggestions", [])
            if suggestions:
                prompt_parts.append(
                    f"Suggestions: {'; '.join(suggestions)}"
                )

        if previous_prompts:
            prompt_parts.append(
                "\nNote: Previous prompts that produced invalid results: "
            )
            for p in previous_prompts[-3:]:
                prompt_parts.append(f"  - {p[:200]}")

        prompt_parts.append(
            "\nMake sure all generated Mermaid diagrams are syntactically valid, "
            "start with the correct diagram type keyword, have balanced brackets, "
            "and follow Mermaid.js best practices."
        )

        return "\n".join(prompt_parts)

    def _validate_single_diagram(
        self,
        diagram: str,
        diagram_type: str,
    ) -> dict[str, Any]:
        """
        Validate a single Mermaid diagram.

        Parameters
        ----------
        diagram : str
            Mermaid diagram code.
        diagram_type : str
            Diagram type.

        Returns
        -------
        dict[str, Any]
            Validation results.
        """
        self.logger.info(
            "Running validation for diagram_type=%s",
            diagram_type,
        )

        self.logger.info(
            "Diagram size=%d chars",
            len(diagram),
        )

        basic_result = self._basic_mermaid_validation(diagram)

        self.logger.info(
            "Basic validation result=%s",
            basic_result,
        )

        if not basic_result["is_valid"]:
            self.logger.warning(
                "Basic Mermaid validation failed: %s",
                basic_result["feedback"],
            )

            return {
                **basic_result,
                "is_valid": False,
                "feedback": basic_result["feedback"],
                "suggestions": [],
            }

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
            + "FEEDBACK: [detailed explanation]\n"
        )

        self.logger.info(
            "Validation prompt length=%d",
            len(validation_prompt),
        )

        try:
            llm_response = self._chat_completion_with_retry(
                messages=[
                    {
                        "role": "user",
                        "content": validation_prompt,
                    }
                ],
                max_tokens=300,
            )

            response_text = (
                llm_response.choices[0].message.content.strip()
                if hasattr(llm_response, "choices")
                else str(llm_response).strip()
            )

            self.logger.info(
                "Raw validator response:\n%s",
                response_text,
            )

            parsed = self._parse_llm_validation(response_text)

            self.logger.info(
                "Parsed validation result=%s",
                parsed,
            )

            return parsed

        except Exception as exc:
            self.logger.error(
                "Validation LLM call failed: %s",
                exc,
                exc_info=True,
            )

            return {
                "is_valid": False,
                "feedback": f"Validation LLM call failed: {exc}",
                "critical_issues": [],
                "warnings": [],
                "suggestions": [],
            }