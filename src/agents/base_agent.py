"""Base agent class for FlowForge agents."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.database.session_manager import SessionManager
from src.logger import setup_logger


class BaseAgent(ABC):
    """Abstract base class for all FlowForge agents."""

    def __init__(self, name: str, session_manager: Optional[SessionManager] = None) -> None:
        """
        Initialize the base agent.

        Parameters
        ----------
        name : str
            Name of the agent.
        session_manager : SessionManager, optional
            Session manager for database persistence.
        """
        self.name = name
        self.session_manager = session_manager
        self.logger = setup_logger()

    @abstractmethod
    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the agent logic.

        Parameters
        ----------
        state : dict[str, Any]
            Current workflow state.

        Returns
        -------
        dict[str, Any]
            Updated workflow state.
        """

    def _save_session_output(
        self,
        output_type: str,
        output_data: dict[str, Any],
        feedback: str | None = None,
        is_valid: bool = True,
    ) -> Optional[int]:
        """Save agent output to session database."""
        if self.session_manager:
            return self.session_manager.save_agent_output(
                agent_name=self.name,
                output_type=output_type,
                output_data=output_data,
                feedback=feedback,
                is_valid=is_valid,
            )
        return None

    def _get_previous_outputs(
        self,
        output_type: str,
    ) -> list[dict[str, Any]]:
        """Get previous outputs of this type from session database."""
        if self.session_manager:
            return self.session_manager.get_previous_outputs(self.name, output_type)
        return []

    def _get_previous_invalid_output(
        self,
        output_type: str,
    ) -> Optional[dict[str, Any]]:
        """Get most recent invalid output from session database."""
        if self.session_manager:
            return self.session_manager.get_previous_invalid_output(self.name, output_type)
        return None

    def _update_state(
        self,
        state: dict[str, Any],
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update workflow state.

        Parameters
        ----------
        state : dict[str, Any]
            Existing workflow state.

        updates : dict[str, Any]
            Values to update.

        Returns
        -------
        dict[str, Any]
            Updated workflow state.
        """
        state.update(updates)
        state["current_agent"] = self.name

        self.logger.info("Agent %s executed", self.name)

        return state
    