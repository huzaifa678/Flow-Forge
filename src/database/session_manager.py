from typing import Any, Optional

from src.database.service import SessionService


class SessionManager:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.service = SessionService()

    def create_session(
        self,
        proposal: str,
        prompt: str,
        project_title: str = "",
        timeline_weeks: int = 12,
        team_size: int = 5,
        tech_stack: list[str] | None = None,
        priority: str = "medium",
        diagram_types: list[str] | None = None,
    ) -> str:
        self.session_id = self.service.create_session(
            proposal=proposal,
            prompt=prompt,
            project_title=project_title,
            timeline_weeks=timeline_weeks,
            team_size=team_size,
            tech_stack=tech_stack,
            priority=priority,
            diagram_types=diagram_types,
        )
        return self.session_id

    def save_agent_output(
        self,
        agent_name: str,
        output_type: str,
        output_data: dict[str, Any],
        feedback: str | None = None,
        is_valid: bool = True,
    ) -> int:
        if not self.session_id:
            raise ValueError("No active session")

        version = 1
        existing = self.service.get_latest_output(
            self.session_id, agent_name, output_type
        )
        if existing:
            version = existing["version"] + 1

        return self.service.save_output(
            session_id=self.session_id,
            agent_name=agent_name,
            output_type=output_type,
            output_data=output_data,
            feedback=feedback,
            is_valid=is_valid,
            version=version,
        )

    def get_previous_outputs(
        self,
        agent_name: str,
        output_type: str,
    ) -> list[dict[str, Any]]:
        if not self.session_id:
            return []
        return self.service.get_outputs(
            self.session_id, agent_name, output_type
        )

    def get_latest_output(
        self,
        agent_name: str,
        output_type: str,
    ) -> Optional[dict[str, Any]]:
        if not self.session_id:
            return None
        return self.service.get_latest_output(
            self.session_id, agent_name, output_type
        )

    def get_previous_invalid_output(
        self,
        agent_name: str,
        output_type: str,
    ) -> Optional[dict[str, Any]]:
        if not self.session_id:
            return None
        return self.service.get_previous_invalid_output(
            self.session_id, agent_name, output_type
        )

    def set_session_id(self, session_id: str) -> None:
        self.session_id = session_id