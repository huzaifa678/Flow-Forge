"""Time Validator Agent for FlowForge - validates time agent outputs.

Validates the timetable, milestones, and Gantt chart produced by TimeAgent.
When validation fails, signals the workflow to route back to TimeAgent for
regeneration with corrective feedback.
"""

from typing import Any, Optional

from src.agents.base_validator_agent import BaseValidatorAgent


class TimeValidatorAgent(BaseValidatorAgent):
    """Validate time agent outputs (timetable, milestones, Gantt charts).

    When validation fails, sets 'route_to' = 'time_agent' in the state
    so the workflow re-routes to the time agent with corrective prompts.
    """

    TIMELINE_VALIDATOR_SYSTEM_PROMPT = """You are a senior project planning expert
who validates project timelines, milestones, and Gantt charts. Your job is to
ensure that:

1. The timeline covers all major project phases (Discovery, Design, Development,
   Testing, Deployment)
2. Milestones are clearly defined with measurable deliverables
3. Parallel work streams are properly identified
4. The Gantt chart uses valid Mermaid gantt syntax
5. Durations are realistic and account for dependencies
6. Buffer time is included for reviews and iterations
7. No critical tasks are missing or improperly sequenced

Be strict in your validation. Flag any timeline that lacks:
- Clear phase boundaries
- Dependency ordering
- Realistic duration estimates
- Identified parallel work streams
- A valid Mermaid gantt diagram

Return structured validation results in the EXACT format specified."""

    def __init__(self, session_manager: Optional[Any] = None) -> None:
        """Initialize time validator agent."""
        super().__init__("time_validator_agent", session_manager=session_manager)

    def _get_model(self) -> str:
        """Return the HuggingFace model identifier."""
        return "Qwen/Qwen2.5-VL-7B-Instruct"

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Validate the time agent's outputs.

        Checks the timetable, milestones, and Gantt chart. If validation fails,
        provides corrective feedback and routes back to the time agent.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing timetable, milestones, etc.

        Returns
        -------
        dict[str, Any]
            Updated workflow state with validation results and routing info.
        """
        try:
            self.logger.info("Starting time output validation.")

            timetable = state.get("timetable", "")
            milestones = state.get("milestones", [])
            parallel_streams = state.get("parallel_work_streams", [])

            validation_results = []
            all_valid = True

            # ---------- Validate timetable ----------
            self.logger.info("Validating timetable.")
            time_result = self._validate_timetable(timetable)
            time_result["field"] = "timetable"
            validation_results.append(time_result)

            if not time_result["is_valid"]:
                all_valid = False
                self.logger.warning(
                    "Timetable validation failed: %s", time_result["feedback"]
                )

            # ---------- Validate milestones ----------
            self.logger.info("Validating milestones.")
            milestone_result = self._validate_milestones(milestones)
            milestone_result["field"] = "milestones"
            validation_results.append(milestone_result)

            if not milestone_result["is_valid"]:
                all_valid = False
                self.logger.warning(
                    "Milestones validation failed: %s", milestone_result["feedback"]
                )

            # ---------- Validate parallel streams ----------
            self.logger.info("Validating parallel work streams.")
            parallel_result = self._validate_parallel_streams(parallel_streams)
            parallel_result["field"] = "parallel_work_streams"
            validation_results.append(parallel_result)

            if not parallel_result["is_valid"]:
                all_valid = False
                self.logger.warning(
                    "Parallel streams validation failed: %s", parallel_result["feedback"]
                )

            # ---------- Validate Gantt chart syntax ----------
            self.logger.info("Validating Gantt chart syntax.")
            gantt_result = self._basic_mermaid_validation(timetable)
            gantt_result["field"] = "gantt_syntax"
            validation_results.append(gantt_result)

            if not gantt_result["is_valid"]:
                all_valid = False
                self.logger.warning(
                    "Gantt chart syntax validation failed: %s", gantt_result["feedback"]
                )

            valid_count = sum(
                1 for r in validation_results if r["is_valid"]
            )

            self.logger.info(
                "Time validation complete | valid=%d/%d",
                valid_count,
                len(validation_results),
            )

            if all_valid:
                return self._update_state(
                    state,
                    {
                        "time_validation_results": validation_results,
                        "time_overall_validation": True,
                        "error": None,
                        "route_to": "plan_agent",
                        "current_agent": "time_validator_agent",
                    },
                )
            else:
                # Build corrective feedback for the time agent
                corrective_prompt = self._build_corrective_prompt(
                    validation_results, timetable
                )

                result_state = self._update_state(
                    state,
                    {
                        "time_validation_results": validation_results,
                        "time_overall_validation": False,
                        "time_valid_count": valid_count,
                        "time_total_count": len(validation_results),
                        "error": (
                            f"Time validation failed: "
                            f"{len(validation_results) - valid_count}/"
                            f"{len(validation_results)} checks failed"
                        ),
                        "corrective_prompt": corrective_prompt,
                        "route_to": "time_agent",
                        "current_agent": "time_validator_agent",
                    },
                )

                self._save_session_output(
                    output_type="time_validation",
                    output_data={
                        "validation_results": validation_results,
                        "overall_valid": False,
                        "valid_count": valid_count,
                        "total_checks": len(validation_results),
                    },
                    feedback=(
                        f"{valid_count}/{len(validation_results)} checks passed. "
                        f"Routing back to time agent for regeneration."
                    ),
                    is_valid=False,
                )

                return result_state
                    

        except Exception as exc:
            error_msg = f"Time validator agent failed: {exc}"

            self.logger.error(
                error_msg,
                exc_info=True,
            )

            return self._update_state(
                state,
                {
                    "error": error_msg,
                    "time_validation_results": [],
                    "time_overall_validation": False,
                    "route_to": "end",
                    "current_agent": "time_validator_agent",
                },
            )

    def _validate_timetable(
        self,
        timetable: str,
    ) -> dict[str, Any]:
        """Validate that the timetable is substantive and well-formed.

        Parameters
        ----------
        timetable : str
            The raw timetable text from the time agent.

        Returns
        -------
        dict[str, Any]
            Validation result.
        """
        if not timetable or not timetable.strip():
            return {
                "is_valid": False,
                "feedback": "Timetable is empty. Must generate a project timeline.",
                "critical_issues": ["No timetable generated"],
                "warnings": [],
                "suggestions": [
                    "Generate a phased timeline with estimated durations for each phase."
                ],
            }

        # Check for minimum substantive content
        lines = [l.strip() for l in timetable.split("\n") if l.strip()]

        if len(lines) < 3:
            return {
                "is_valid": False,
                "feedback": (
                    "Timetable is too sparse. Expected at least 3 lines of content "
                    "(phases, tasks, or milestones)."
                ),
                "critical_issues": ["Insufficient timetable detail"],
                "warnings": [],
                "suggestions": [
                    "Include multiple project phases with task breakdowns."
                ],
            }

        # Check for phase-like content
        phase_keywords = [
            "phase",
            "design",
            "development",
            "discovery",
            "testing",
            "deployment",
            "planning",
            "implementation",
            "milestone",
        ]
        has_phases = any(
            kw in timetable.lower() for kw in phase_keywords
        )

        if not has_phases:
            return {
                "is_valid": False,
                "feedback": (
                    "Timetable appears to lack structured phases. "
                    "Expected clear project phases (e.g., Discovery, Design, "
                    "Development, Testing, Deployment)."
                ),
                "critical_issues": ["No identifiable project phases in timetable"],
                "warnings": [],
                "suggestions": [
                    "Organize the timeline into clear phases with entry/exit criteria."
                ],
            }

        return {
            "is_valid": True,
            "feedback": "Timetable structure is adequate.",
            "critical_issues": [],
            "warnings": [],
            "suggestions": [],
        }

    def _validate_milestones(
        self,
        milestones: list,
    ) -> dict[str, Any]:
        """Validate that milestones are meaningful.

        Parameters
        ----------
        milestones : list
            List of milestone strings.

        Returns
        -------
        dict[str, Any]
            Validation result.
        """
        if not milestones or len(milestones) == 0:
            return {
                "is_valid": False,
                "feedback": "No milestones generated.",
                "critical_issues": ["Missing milestones"],
                "warnings": [],
                "suggestions": [
                    "Define at least 3-5 milestones with clear deliverables."
                ],
            }

        # Filter out trivially short milestones
        substantive = [
            m for m in milestones
            if isinstance(m, str) and len(m.strip()) > 5
        ]

        if len(substantive) < 2:
            return {
                "is_valid": False,
                "feedback": (
                    "Too few substantive milestones. "
                    f"Found {len(substantive)} with meaningful content."
                ),
                "critical_issues": ["Insufficient milestones"],
                "warnings": [],
                "suggestions": [
                    "Add milestone markers for each major phase transition."
                ],
            }

        return {
            "is_valid": True,
            "feedback": f"{len(substantive)} substantive milestones found.",
            "critical_issues": [],
            "warnings": [],
            "suggestions": [],
        }

    def _validate_parallel_streams(
        self,
        parallel_streams: list,
    ) -> dict[str, Any]:
        """Validate that parallel work streams are identified.

        Parameters
        ----------
        parallel_streams : list
            List of parallel work stream strings.

        Returns
        -------
        dict[str, Any]
            Validation result.
        """
        if not parallel_streams or len(parallel_streams) == 0:
            return {
                "is_valid": False,
                "feedback": "No parallel work streams identified.",
                "critical_issues": ["Missing parallel work stream analysis"],
                "warnings": [],
                "suggestions": [
                    "Identify tasks that can run concurrently to optimize project duration."
                ],
            }

        # Filter out trivially short streams
        substantive = [
            s for s in parallel_streams
            if isinstance(s, str) and len(s.strip()) > 5
        ]

        if len(substantive) < 1:
            return {
                "is_valid": False,
                "feedback": "Parallel work streams lack substantive content.",
                "critical_issues": [],
                "warnings": ["Parallel streams are too vague"],
                "suggestions": [
                    "Name specific work streams (e.g., 'Frontend Development', 'API Design')."
                ],
            }

        return {
            "is_valid": True,
            "feedback": f"{len(substantive)} parallel work streams identified.",
            "critical_issues": [],
            "warnings": [],
            "suggestions": [],
        }

    def _build_corrective_prompt(
        self,
        validation_results: list[dict],
        timetable: str,
    ) -> str:
        """Build a corrective prompt for the time agent.

        Parameters
        ----------
        validation_results : list[dict]
            Results from all validation checks.
        timetable : str
            The original timetable output.

        Returns
        -------
        str
            Corrective prompt to send to the time agent on re-routing.
        """
        parts = [
            "IMPORTANT: The previously generated timeline failed validation. "
            "Please regenerate the timeline addressing ALL of the following issues:"
        ]

        for result in validation_results:
            field = result.get("field", "unknown")
            is_valid = result.get("is_valid", True)

            if is_valid:
                continue

            feedback = result.get("feedback", "")
            critical = result.get("critical_issues", [])
            suggestions = result.get("suggestions", [])

            parts.append(f"\n--- {field.upper()} ---")
            if feedback:
                parts.append(f"Issue: {feedback}")
            for issue in critical:
                parts.append(f"CRITICAL: {issue}")
            for suggestion in suggestions:
                parts.append(f"Fix: {suggestion}")

        parts.append(
            "\nThe generated timeline MUST include:\n"
            "1. At least 4 clearly named project phases\n"
            "2. At least 4 milestones with measurable deliverables\n"
            "3. 2-4 identified parallel work streams\n"
            "4. A valid Mermaid gantt chart with proper syntax\n"
            "5. Realistic duration estimates with buffer time\n"
            "6. Clear task dependencies and sequencing\n"
        )

        return "\n".join(parts)