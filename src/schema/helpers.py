"""Helper functions to format workflow results into API responses."""
from typing import Any


def format_workflow_response(result: dict[str, Any]) -> dict[str, Any]:
    """Format the raw workflow state into a structured API response dict.

    This avoids Pydantic validation issues with nested objects in tests
    while still producing a clean, serializable response structure.
    """
    status = "success" if not result.get("error") else "failed"

    # If we have diagrams but some failed, it's partial
    total = result.get("diagram_count", len(result.get("diagrams", [])))
    valid = result.get("valid_diagram_count", 0)

    if total > 0 and valid < total:
        status = "partial"
    elif total == 0 and result.get("error"):
        status = "failed"

    # Format timeline output
    timeline = None
    if result.get("timetable"):
        timeline = {
            "milestones": result.get("milestones", []),
            "parallel_work_streams": result.get("parallel_work_streams", []),
            "gantt_chart": (
                result.get("timetable")
                if "gantt" in result.get("timetable", "").lower()
                else None
            ),
            "raw_timetable": result["timetable"],
        }

    return {
        "status": status,
        "proposal_summary": result.get(
            "proposal_summary", result.get("proposal", "")[:200]
        ),
        "optimized_prompt": result.get("optimized_prompt"),
        "timeline": timeline,
        "plan": result.get("plan"),
        "diagrams": result.get("diagrams", []),
        "valid_diagram_count": valid,
        "total_diagram_count": total,
        "overall_validation": result.get("overall_validation", False),
        "validation_feedback": result.get("feedback"),
        "current_agent": result.get("current_agent"),
        "error": result.get("error"),
    }