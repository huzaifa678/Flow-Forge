"""Time Agent for FlowForge."""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_endpoint import (
    HuggingFaceEndpoint,
)

from src.agents.base_agent import BaseAgent
from src.config import Config


class TimeAgent(BaseAgent):
    """Generate project timetables from user prompts."""

    def __init__(self) -> None:
        """Initialize the time agent."""
        super().__init__("time_agent")

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
            repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.7,
            huggingfacehub_api_token=hf_token,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate timetable from prompt.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state.

        Returns
        -------
        dict[str, Any]
            Updated workflow state.
        """
        try:
            prompt = state.get("prompt", "")

            if not prompt:
                error_msg = "No prompt provided for time agent."

                self.logger.error(error_msg)

                return self._update_state(
                    state,
                    {"error": error_msg},
                )

            self.logger.info(
                "Generating timetable for prompt."
            )

            timetable_template = """
            You are a project planning expert.

            Given the following user proposal,
            create a detailed timetable with:

            1. Project phases
            2. Key milestones
            3. Estimated durations
            4. Parallel work streams
            5. Dependencies

            User Prompt:
            {prompt}
            """

            prompt_template = PromptTemplate(
                input_variables=["prompt"],
                template=timetable_template,
            )

            formatted_prompt = prompt_template.format(
                prompt=prompt
            )

            timetable = self.llm.invoke(formatted_prompt)

            self.logger.info(
                "Timetable generated successfully."
            )

            return self._update_state(
                state,
                {
                    "timetable": timetable.strip(),
                    "error": None,
                    "current_agent": "time_agent",
                },
            )

        except Exception as exc:
            error_msg = (
                f"Time agent failed: {exc}"
            )

            self.logger.error(
                error_msg,
                exc_info=True,
            )

            return self._update_state(
                state,
                {
                    "error": "Failed to generate timetable",
                    "timetable": None,
                    "current_agent": "time_agent",
                },
            )
        