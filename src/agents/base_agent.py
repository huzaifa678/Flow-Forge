"""Base agent class for FlowForge agents."""

from abc import ABC, abstractmethod
from typing import Any

from src.logger import setup_logger


class BaseAgent(ABC):
    """Abstract base class for all FlowForge agents."""

    def __init__(self, name: str) -> None:
        """
        Initialize the base agent.

        Parameters
        ----------
        name : str
            Name of the agent.
        """
        self.name = name
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
    