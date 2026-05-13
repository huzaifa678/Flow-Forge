"""Helper functions to format workflow results into API responses."""
from typing import Any


def format_workflow_response(result: dict[str, Any]) -> dict[str, Any]:
    """Format workflow state into API response with image data included."""
    has_error = bool(result.get("error"))

    total = result.get("diagram_count", len(result.get("diagrams", [])))
    valid = result.get("valid_diagram_count", 0)

    if total > 0 and valid < total:
        status = "partial"
    elif total == 0 and has_error:
        status = "failed"
    else:
        status = "success"

    # Timeline formatting
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

    diagrams_out = []
    for d in result.get("diagrams", []):
        diagrams_out.append({
            "diagram_type": d.get("diagram_type"),
            "mermaid_code": d.get("mermaid_code"),
            "title": d.get("title"),
            "description": d.get("description"),
            "is_valid": d.get("is_valid", False),
            "validation_feedback": d.get("validation_feedback"),
            "image_data": d.get("image_data"),
        })

    return {
        "status": status,
        "proposal_summary": result.get(
            "proposal_summary", result.get("proposal", "")[:200]
        ),
        "optimized_prompt": result.get("optimized_prompt"),
        "timeline": timeline,
        "plan": result.get("plan"),
        "diagrams": diagrams_out,
        "valid_diagram_count": valid,
        "total_diagram_count": total,
        "overall_validation": result.get("overall_validation", False),
        "validation_feedback": result.get("feedback"),
        "current_agent": result.get("current_agent"),
        "error": result.get("error"),
    }