"""Image Generator Agent for FlowForge."""

from typing import Any

from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

from src.agents.base_agent import BaseAgent
from src.config import Config


class ImageGeneratorAgent(BaseAgent):
    """Generate Mermaid diagrams from project plans."""

    def __init__(self) -> None:
        """Initialize the image generator agent."""
        super().__init__("image_agent")
        self.llm: HuggingFaceEndpoint | None = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize Hugging Face LLM."""
        if not Config.HF_TOKEN:
            raise ValueError("HF_TOKEN must be set.")

        self.llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1",
            task="text-generation",
            max_new_tokens=800,
            temperature=0.3,
            huggingfacehub_api_token=Config.HF_TOKEN,
        )

    def _extract_text(self, response: Any) -> str:
        """Safely extract text from LLM response."""
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        # HuggingFace sometimes returns objects with .content
        return getattr(response, "content", str(response))

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate Mermaid diagram from plan.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing plan.

        Returns
        -------
        dict[str, Any]
            Updated workflow state.
        """
        try:
            plan = state.get("plan")

            if not plan:
                return self._update_state(
                    state,
                    {
                        "error": "Failed to generate image: no plan provided",
                        "diagram": None,
                    },
                )

            prompt_template = PromptTemplate(
                input_variables=["plan"],
                template="""
You are an expert Mermaid diagram generator.

Return ONLY valid Mermaid syntax.

Project Plan:
{plan}

Rules:
- Choose Gantt OR flowchart
- Include dependencies
- Include parallel tasks
- No explanation text
- No markdown fences
""",
            )

            prompt = prompt_template.format(plan=plan)

            response = self.llm.invoke(prompt)
            diagram = self._extract_text(response).strip()

            # cleanup markdown fences if model ignores instructions
            if diagram.startswith("```"):
                diagram = diagram.replace("```mermaid", "").replace("```", "").strip()

            if not diagram or "graph" not in diagram and "gantt" not in diagram:
                return self._update_state(
                    state,
                    {
                        "error": "No Mermaid diagram provided.",
                        "diagram": None,
                    },
                )

            return self._update_state(
                state,
                {
                    "diagram": diagram,   
                    "error": None,
                },
            )

        except Exception as exc:
            return self._update_state(
                state,
                {
                    "error": f"Failed to generate image: {exc}",
                    "diagram": None,
                },
            )
        