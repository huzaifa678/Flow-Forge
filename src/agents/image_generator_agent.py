"""Image Generator Agent for FlowForge - supports multiple diagram types."""

from typing import Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from src.agents.base_agent import BaseAgent
from src.config import Config
from src.schemas.request import DiagramType


class ImageGeneratorAgent(BaseAgent):
    """Generate Mermaid diagrams from project plans - supports multiple diagram types."""

    DIAGRAM_TEMPLATES = {
        DiagramType.WORKFLOW: {
            "system_role": (
                "You are an expert workflow diagram designer. "
                "Create a comprehensive workflow diagram that shows the "
                "end-to-end process flow, decision points, and task sequences."
            ),
            "requirements": (
                "- Use flowchart syntax with clear swimlanes or sequential flow\n"
                "- Include decision diamonds for conditional paths\n"
                "- Show parallel processes where applicable\n"
                "- Label all transitions and steps clearly\n"
                "- Start and end nodes must be present"
            ),
        },
        DiagramType.CI_CD: {
            "system_role": (
                "You are an expert CI/CD pipeline architect. "
                "Create a detailed CI/CD pipeline diagram showing build, "
                "test, staging, and deployment stages."
            ),
            "requirements": (
                "- Show all pipeline stages (build, test, security scan, deploy)\n"
                "- Include branching strategies (feature, develop, main)\n"
                "- Show artifact flows and triggers\n"
                "- Include rollback mechanisms\n"
                "- Use gantt or flowchart syntax"
            ),
        },
        DiagramType.SYSTEM_DESIGN: {
            "system_role": (
                "You are an expert system design architect. "
                "Create a comprehensive system design diagram showing "
                "all components, services, and their interactions."
            ),
            "requirements": (
                "- Show all microservices/services and their relationships\n"
                "- Include databases, caches, and external services\n"
                "- Show data flow directions\n"
                "- Include load balancers and API gateways\n"
                "- Use flowchart syntax with clear component boundaries"
            ),
        },
        DiagramType.FLOWCHART: {
            "system_role": (
                "You are an expert business process analyst. "
                "Create a clear flowchart showing business logic and process flows."
            ),
            "requirements": (
                "- Use standard flowchart symbols\n"
                "- Include start/end terminators\n"
                "- Show decision points with clear yes/no paths\n"
                "- Group related processes into subgraphs\n"
                "- Keep flow left-to-right or top-to-bottom"
            ),
        },
        DiagramType.ARCHITECTURE: {
            "system_role": (
                "You are an enterprise solution architect. "
                "Create a comprehensive architecture diagram showing the "
                "overall system architecture, layers, and technology choices."
            ),
            "requirements": (
                "- Show layered architecture (presentation, business, data)\n"
                "- Include technology stack annotations\n"
                "- Show network zones and boundaries\n"
                "- Include caching layers and message queues\n"
                "- Use flowchart syntax with clear layer separation"
            ),
        },
        DiagramType.GANTT: {
            "system_role": (
                "You are an expert project manager. "
                "Create a detailed Gantt chart showing project timeline, "
                "milestones, dependencies, and parallel work streams."
            ),
            "requirements": (
                "- Use gantt syntax exclusively\n"
                "- Include project title and date sections\n"
                "- Show task dependencies (after, before)\n"
                "- Identify parallel work streams\n"
                "- Include milestones and critical path"
            ),
        },
    }

    def __init__(self) -> None:
        """Initialize the image generator agent."""
        super().__init__("image_agent")
        self.llm: Optional[HuggingFaceEndpoint] = None
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize Hugging Face LLM."""
        if not Config.HF_TOKEN:
            raise ValueError("HF_TOKEN must be set.")

        self.llm = HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1",
            task="text-generation",
            max_new_tokens=1200,
            temperature=0.3,
            huggingfacehub_api_token=Config.HF_TOKEN,
        )

    def _extract_text(self, response: Any) -> str:
        """Safely extract text from LLM response."""
        if response is None:
            return ""

        if isinstance(response, str):
            return response

        return getattr(response, "content", str(response))

    def _build_diagram_prompt(
        self,
        plan: str,
        diagram_type: DiagramType,
        timeline: Optional[str] = None,
    ) -> str:
        """Build an optimized prompt for a specific diagram type."""
        import json

        template_config = self.DIAGRAM_TEMPLATES.get(
            diagram_type, self.DIAGRAM_TEMPLATES[DiagramType.FLOWCHART]
        )

        prompt_template = PromptTemplate(
            input_variables=["plan", "timeline", "diagram_type_name"],
            template="""
{system_role}

Rules:
{rules}

Plan:
{plan}

Timeline Reference:
{timeline}

Generate the complete {diagram_type_name} diagram in valid Mermaid syntax.
Return ONLY the Mermaid code. No explanations. No markdown fences.
""".format(
                system_role=template_config["system_role"],
                rules=template_config["requirements"],
                plan=plan,
                timeline=timeline or "No timeline provided",
                diagram_type_name=diagram_type.value.replace("_", " ").title(),
            ),
        )
        return prompt_template.format(
            plan=plan,
            timeline=timeline or "",
            diagram_type_name=diagram_type.value.replace("_", " ").title(),
        )

    def _parse_diagram(self, response: str) -> str:
        """Clean and extract Mermaid diagram from LLM response."""
        diagram = self._extract_text(response).strip()

        # Remove markdown fences if model ignores instructions
        if diagram.startswith("```"):
            diagram = diagram.replace("```mermaid", "").replace("```", "").strip()

        return diagram

    def _validate_diagram(self, diagram: str) -> bool:
        """Basic validation of generated Mermaid diagram."""
        if not diagram or len(diagram.strip()) < 10:
            return False

        valid_starts = [
            "gantt",
            "flowchart",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram",
            "erDiagram",
            "graph",
        ]

        diagram_lower = diagram.lower().strip()
        return any(diagram_lower.startswith(v.lower()) for v in valid_starts)

    def generate_diagram(
        self,
        plan: str,
        diagram_type: DiagramType,
        timeline: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate a single Mermaid diagram of the specified type.

        Args:
            plan: The project plan to base the diagram on.
            diagram_type: The type of diagram to generate.
            timeline: Optional timeline reference for context.

        Returns:
            Dict containing diagram data and metadata.
        """
        try:
            if not plan:
                return {
                    "diagram_type": diagram_type.value,
                    "mermaid_code": None,
                    "title": diagram_type.value.replace("_", " ").title(),
                    "description": None,
                    "is_valid": False,
                    "validation_feedback": "No plan provided",
                    "error": "No plan provided",
                }

            prompt = self._build_diagram_prompt(plan, diagram_type, timeline)
            response = self.llm.invoke(prompt)
            diagram = self._parse_diagram(response)

            if not self._validate_diagram(diagram):
                return {
                    "diagram_type": diagram_type.value,
                    "mermaid_code": None,
                    "title": diagram_type.value.replace("_", " ").title(),
                    "description": None,
                    "is_valid": False,
                    "validation_feedback": "No valid Mermaid diagram generated",
                    "error": "No valid diagram generated",
                }

            return {
                "diagram_type": diagram_type.value,
                "mermaid_code": diagram,
                "title": diagram_type.value.replace("_", " ").title(),
                "description": f"Auto-generated {diagram_type.value} diagram",
                "is_valid": True,
                "validation_feedback": "Diagram generated successfully",
                "error": None,
            }

        except Exception as exc:
            return {
                "diagram_type": diagram_type.value,
                "mermaid_code": None,
                "title": diagram_type.value.replace("_", " ").title(),
                "description": None,
                "is_valid": False,
                "validation_feedback": None,
                "error": f"Failed to generate {diagram_type.value} diagram: {exc}",
            }

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate all requested Mermaid diagrams from plan.

        Parameters
        ----------
        state : dict[str, Any]
            Workflow state containing plan, timeline, and diagram types.

        Returns
        -------
        dict[str, Any]
            Updated workflow state with all generated diagrams.
        """
        try:
            plan = state.get("plan")
            timeline = state.get("timetable")
            diagram_types_raw = state.get("diagram_types", [DiagramType.WORKFLOW])

            if not plan:
                return self._update_state(
                    state,
                    {
                        "error": "Failed to generate images: no plan provided",
                        "diagrams": [],
                        "current_agent": "image_agent",
                    },
                )

            # Parse diagram types
            if isinstance(diagram_types_raw, list):
                if all(isinstance(d, str) for d in diagram_types_raw):
                    diagram_types = [
                        DiagramType(d) if isinstance(d, str) else d
                        for d in diagram_types_raw
                    ]
                else:
                    diagram_types = diagram_types_raw
            elif isinstance(diagram_types_raw, str):
                diagram_types = [DiagramType(diagram_types_raw)]
            else:
                diagram_types = [DiagramType.WORKFLOW]

            self.logger.info(
                "Generating %d diagrams: %s",
                len(diagram_types),
                [dt.value for dt in diagram_types],
            )

            # Generate each diagram
            diagrams = []
            for diagram_type in diagram_types:
                diagram_result = self.generate_diagram(
                    plan=plan,
                    diagram_type=diagram_type,
                    timeline=timeline,
                )
                diagrams.append(diagram_result)

            valid_count = sum(1 for d in diagrams if d.get("is_valid"))
            self.logger.info(
                "Generated %d/%d valid diagrams", valid_count, len(diagrams)
            )

            return self._update_state(
                state,
                {
                    "diagrams": diagrams,
                    "diagram_count": len(diagrams),
                    "valid_diagram_count": valid_count,
                    "error": None if valid_count > 0 else "No valid diagrams generated",
                    "current_agent": "image_agent",
                },
            )

        except Exception as exc:
            return self._update_state(
                state,
                {
                    "error": f"Image generator agent failed: {exc}",
                    "diagrams": [],
                    "diagram_count": 0,
                    "valid_diagram_count": 0,
                    "current_agent": "image_agent",
                },
            )