"""Validator Agent for FlowForge - validates Mermaid diagrams with refined checks."""

import re
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
        return "Qwen/Qwen2.5-Coder-32B-Instruct"

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
                elif result.get("suggestions"):
                    # Diagram is valid but has quality suggestions — collect for improvement
                    invalid_diagrams.append({
                        "index": idx,
                        "diagram_type": diagram_type,
                        "feedback": "Quality improvement suggested.",
                        "critical_issues": [],
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

            retry_count = state.get("validation_retry_count", 0)

            # Diagrams with quality suggestions get one improvement pass (retry_count == 0 only)
            has_quality_suggestions = bool(invalid_diagrams) and all_valid
            should_improve = has_quality_suggestions and retry_count == 0

            if all_valid and not should_improve:
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
                        "validation_retry_count": retry_count,
                    },
                )
            elif not all_valid or should_improve:
                improvement_prompt = self._build_improvement_prompt(
                    invalid_diagrams, previous_prompts
                )
                result_state = self._update_state(
                    state,
                    {
                        "validation_results": validation_results,
                        "overall_validation": all_valid,
                        "valid_count": valid_count,
                        "total_count": len(validation_results),
                        "error": (
                            None if all_valid else
                            f"{len([d for d in invalid_diagrams if not next((r for r in validation_results if r.get('diagram_type') == d['diagram_type']), {}).get('is_valid', True)])}/{len(validation_results)} diagrams failed validation"
                        ),
                        "improvement_prompt": improvement_prompt,
                        "failed_diagrams": invalid_diagrams,
                        "route_to": "image_agent",
                        "current_agent": "validator_agent",
                        "validation_retry_count": retry_count + 1,
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
        """Build improvement feedback for the image generator.

        Syntax failures (critical_issues) are hard requirements — must be fixed.
        Quality suggestions are advisory — improve if possible.
        """
        syntax_failures = [d for d in invalid_diagrams if d.get("critical_issues")]
        quality_suggestions = [d for d in invalid_diagrams if not d.get("critical_issues") and d.get("suggestions")]

        prompt_parts = []

        if syntax_failures:
            prompt_parts.append(
                "SYNTAX FIXES REQUIRED — regenerate these diagrams with the exact fixes below:"
            )
            for diag in syntax_failures:
                diag_type = diag.get("diagram_type", "unknown")
                prompt_parts.append(f"\n[{diag_type}] Fix: {diag.get('feedback', '')}")
                for issue in diag.get("critical_issues", []):
                    prompt_parts.append(f"  - {issue}")

        if quality_suggestions:
            prompt_parts.append(
                "\nADVISORY IMPROVEMENTS — apply these to improve clarity and readability:"
            )
            for diag in quality_suggestions:
                diag_type = diag.get("diagram_type", "unknown")
                for sug in diag.get("suggestions", []):
                    prompt_parts.append(f"  [{diag_type}] {sug}")

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
                "critical_issues": [basic_result["feedback"]],
                "warnings": [],
            }

        # Syntax check passed — LLM is advisory only, never overrides is_valid.
        quality_prompt = (
            "You are a diagram quality adviser. The Mermaid diagram below is syntactically valid.\n"
            "Your ONLY job is to suggest improvements to clarity, readability, and audience-appropriateness.\n"
            "You are an ADVISER — you do not block or reject diagrams.\n\n"
            "Advise on:\n"
            "- Whether the diagram is too cluttered (>15 nodes) and could be simplified\n"
            "- Whether backward/loop edges are causing visual crossing lines — suggest removing them for cleaner flow\n"
            "- Whether the content matches the audience (stakeholder = business language only, engineer = technical depth)\n"
            "- Whether subgraphs would improve grouping and reduce crossed arrows\n"
            "- Whether labels are clear and concise\n\n"
            "Do NOT flag:\n"
            "- Syntax correctness (already validated)\n"
            "- Missing technical detail\n"
            "- Stylistic preferences with no readability impact\n\n"
            f"Diagram type: {diagram_type}\n\n"
            f"Mermaid Diagram:\n```\n{diagram}\n```\n\n"
            "Return ONLY this format:\n"
            "HAS_ISSUES: true/false\n"
            "SUGGESTIONS: [list of specific advisory improvements, or None]\n"
        )

        try:
            llm_response = self._chat_completion_with_retry(
                messages=[{"role": "user", "content": quality_prompt}],
                max_tokens=200,
            )
            response_text = (
                llm_response.choices[0].message.content.strip()
                if hasattr(llm_response, "choices")
                else str(llm_response).strip()
            )
            self.logger.info("Quality review response:\n%s", response_text)

            suggestions = []
            has_issues = False

            has_match = re.search(r"HAS_ISSUES:\s*(true|false)", response_text, re.I)
            if has_match:
                has_issues = has_match.group(1).lower() == "true"

            sug_match = re.search(r"SUGGESTIONS:\s*(.+)", response_text, re.I | re.S)
            if sug_match and sug_match.group(1).strip().lower() not in ("none", "[]"):
                suggestions = [
                    s.strip()
                    for s in re.split(r"[\n,]", sug_match.group(1))
                    if s.strip() and s.strip() not in ("-", "[]", "None")
                ]

            return {
                "is_valid": True,  # always True — basic validation already passed
                "feedback": "Basic validation passed.",
                "critical_issues": [],
                "warnings": suggestions if has_issues else [],
                "suggestions": suggestions,
            }

        except Exception as exc:
            self.logger.warning("Quality review LLM call failed: %s", exc)
            return {
                "is_valid": True,
                "feedback": "Basic validation passed.",
                "critical_issues": [],
                "warnings": [],
                "suggestions": [],
            }