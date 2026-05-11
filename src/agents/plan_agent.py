"""Plan Agent for FlowForge."""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_endpoint import (
    HuggingFaceEndpoint,
)

from src.agents.base_agent import BaseAgent
from src.config import Config


class PlanAgent(BaseAgent):
    """Generate detailed project plans from timetables."""

    def __init__(self) -> None:
        """Initialize the plan agent."""
        super().__init__("plan_agent")

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
            repo_id="deepseek-ai/DeepSeek-R1",
            task="text-generation",
            max_new_tokens=1024,
            temperature=0.6,
            huggingfacehub_api_token=hf_token,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a detailed project plan.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing timetable.

        Returns
        -------
        dict[str, Any]
            Updated workflow state.
        """
        try:
            timetable = state.get("timetable", "")

            if not timetable:
                return self._update_state(
                    state,
                    {
                        "error": "No timetable provided for plan agent.",
                        "plan": None,
                        "current_agent": "plan_agent",
                    },
                )

            self.logger.info(
                "Generating detailed project plan."
            )

            plan_template = """
            You are a detailed project planner.

            Create a comprehensive project plan
            from the following timetable.

            Include:

            1. Task breakdown
            2. Resource allocation
            3. Risk mitigation
            4. Communication structure
            5. QA procedures
            6. Budget estimation

            Timetable:
            {timetable}
            """

            prompt_template = PromptTemplate(
                input_variables=["timetable"],
                template=plan_template,
            )

            formatted_prompt = prompt_template.format(
                timetable=timetable
            )

            plan = self.llm.invoke(formatted_prompt)

            self.logger.info(
                "Plan generated successfully."
            )

            return self._update_state(
                state,
                {
                    "plan": plan.strip(),
                    "error": None,
                    "current_agent": "plan_agent",
                },
            )

        except Exception as exc:
            error_msg = (
                f"Plan agent failed: {exc}"
            )

            self.logger.error(
                error_msg,
                exc_info=True,
            )

            return self._update_state(
                state,
                {
                    "error": "Failed to generate plan",
                    "plan": None,
                    "current_agent": "plan_agent",
                },
            )
        