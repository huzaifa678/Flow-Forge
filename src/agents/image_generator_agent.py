"""Image Generator Agent for FlowForge - supports multiple diagram types."""

import base64
import re
import time
from typing import Any, Optional

import requests
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.prompts import PromptTemplate

from src.agents.base_agent import BaseAgent
from src.config import Config
from src.schemas.request import DiagramType

try:
    from mermaid import Mermaid
    MERMAID_AVAILABLE = True
except ImportError:
    MERMAID_AVAILABLE = False
    Mermaid = None


class ImageGeneratorAgent(BaseAgent):
    """Generate Mermaid diagrams from project plans."""

    # Diagram types relevant to each audience
    STAKEHOLDER_DIAGRAM_TYPES = {DiagramType.FLOWCHART, DiagramType.GANTT, DiagramType.ARCHITECTURE}

    # Stakeholder-focused templates: business language, no technical jargon
    STAKEHOLDER_DIAGRAM_TEMPLATES = {
        DiagramType.FLOWCHART: {
            "system_role": (
                "You are a business analyst presenting project phases to executive stakeholders. "
                "Use plain business language. FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT include CI/CD pipelines, API gateways, microservices, "
                "infrastructure components, tech stack names, or any engineering implementation detail."
            ),
            "requirements": (
                "- Use graph TD syntax\n"
                "- Show high-level business phases and decision gates — NO technical jargon\n"
                "- Label decisions with business outcomes (Approved / Needs Revision)\n"
                "- Use {} for approvals/decisions, [] for phases\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN nodes: anything referencing Docker, Kubernetes, APIs, databases by name, "
                "CI/CD, deployments, or infrastructure\n"
                "- Example:\n"
                "graph TD\n"
                "    Discovery[Discovery & Planning] --> Approval1{Stakeholder Approval}\n"
                "    Approval1 -->|Approved| Design[Design & Prototype]\n"
                "    Approval1 -->|Needs Revision| Discovery\n"
                "    Design --> Review1{Design Review}\n"
                "    Review1 -->|Approved| Build[Build & Deliver]\n"
                "    Review1 -->|Needs Revision| Design\n"
                "    Build --> UAT[User Acceptance Testing]\n"
                "    UAT --> Launch{Ready to Launch?}\n"
                "    Launch -->|Yes| GoLive[Go Live]\n"
                "    Launch -->|No| Build\n"
                "    GoLive --> Support[Ongoing Support]"
            ),
        },
        DiagramType.GANTT: {
            "system_role": (
                "You are a project manager presenting a milestone timeline to business stakeholders. "
                "Show only key milestones and phases. FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT include sprint names, deployment pipelines, infra setup tasks, "
                "or any technical engineering work items — only business-visible milestones and phases."
            ),
            "requirements": (
                "- You MUST start with 'gantt' on the first line\n"
                "- You MUST include 'title' and 'dateFormat YYYY-MM-DD'\n"
                "- Use 'section' for phases; keep task names business-friendly\n"
                "- Mark final deliverables as milestones\n"
                "- FORBIDDEN tasks: CI/CD setup, infrastructure provisioning, technical spike, "
                "docker build, kubernetes config, or any engineering-internal task\n"
                "- Example:\n"
                "gantt\n"
                "    title Project Milestones\n"
                "    dateFormat YYYY-MM-DD\n"
                "    section Initiation\n"
                "    Project Kickoff :a1, 2026-05-14, 5d\n"
                "    section Planning\n"
                "    Business Requirements :a2, after a1, 7d\n"
                "    section Delivery\n"
                "    Build & Test :a3, after a2, 21d\n"
                "    section Launch\n"
                "    Go Live :milestone, after a3, 0d"
            ),
        },
        DiagramType.ARCHITECTURE: {
            "system_role": (
                "You are presenting a high-level solution overview to non-technical stakeholders. "
                "Use business capability names, not technology names. FOCUS ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT include Docker, Kubernetes, Redis, Kafka, PostgreSQL, API Gateway, "
                "microservice names, infrastructure layers, or any technology product names."
            ),
            "requirements": (
                "- Use graph TD syntax with subgraph to show capability areas\n"
                "- Label components with business capabilities (e.g. 'Customer Portal', 'Reporting', 'Data Storage')\n"
                "- Avoid ALL acronyms and technical terms\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN nodes: anything named after a technology product, protocol, or infrastructure service\n"
                "- Example:\n"
                "graph TD\n"
                "    subgraph Users[Users]\n"
                "        Customers[Customers]\n"
                "        Admins[Administrators]\n"
                "    end\n"
                "    subgraph Platform[Platform]\n"
                "        Portal[Customer Portal]\n"
                "        Reports[Reporting Dashboard]\n"
                "    end\n"
                "    subgraph Backend[Business Logic]\n"
                "        Processing[Order Processing]\n"
                "        Notifications[Notifications]\n"
                "    end\n"
                "    subgraph Storage[Data]\n"
                "        Records[Business Records]\n"
                "        Analytics[Analytics Store]\n"
                "    end\n"
                "    Customers --> Portal\n"
                "    Admins --> Reports\n"
                "    Portal --> Processing\n"
                "    Processing --> Records\n"
                "    Processing --> Notifications\n"
                "    Records --> Analytics\n"
                "    Analytics --> Reports"
            ),
        },
    }

    DIAGRAM_TEMPLATES = {
        DiagramType.WORKFLOW: {
            "system_role": (
                "You are an expert workflow diagram designer. REMEMBER generate the mermaid code by FOCUSING ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT produce a high-level business-only phase diagram — show the actual technical workflow with real system steps, services, and decision logic."
            ),
            "requirements": (
                "- Use graph TD syntax\n"
                "- Show the actual technical project workflow with decision points and real system steps\n"
                "- Use {} for decisions, [] for processes\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase (no spaces)\n"
                "- FORBIDDEN: Do NOT use vague phase labels like 'Phase 1', 'Delivery', 'Go Live' — use real technical process names\n"
                "- Example:\n"
                "graph TD\n"
                "    Start[Project Start] --> Req[Requirements Analysis]\n"
                "    Req --> Design[System Design]\n"
                "    Design --> Dev{Development Ready?}\n"
                "    Dev -->|Yes| Backend[Backend Dev]\n"
                "    Dev -->|No| Design\n"
                "    Backend --> Frontend[Frontend Dev]\n"
                "    Frontend --> QA[QA Testing]\n"
                "    QA --> Deploy[Deployment]\n"
                "    Deploy --> End[Project Complete]"
            ),
        },
        DiagramType.CI_CD: {
            "system_role": (
                "You are an expert CI/CD architect. REMEMBER generate the mermaid code by FOCUSING ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT replace CI/CD stages with business phase names — this diagram is for engineers and must show real pipeline stages."
            ),
            "requirements": (
                "- Use graph TD syntax\n"
                "- Show actual CI/CD pipeline stages: code commit, build, test, deploy\n"
                "- Include branch strategy and rollback paths\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN: Do NOT use business language like 'Approval', 'Launch', 'Delivery Phase' — use real CI/CD terminology\n"
                "- Example:\n"
                "graph TD\n"
                "    Commit[Code Commit] --> Build[Docker Build]\n"
                "    Build --> UnitTest[Unit Tests]\n"
                "    UnitTest --> IntTest[Integration Tests]\n"
                "    IntTest --> Pass{Tests Pass?}\n"
                "    Pass -->|Yes| Staging[Deploy to Staging]\n"
                "    Pass -->|No| Rollback[Rollback]\n"
                "    Staging --> Approve{Manual Approval}\n"
                "    Approve -->|Yes| Prod[Deploy to Production]\n"
                "    Approve -->|No| Rollback"
            ),
        },
        DiagramType.SYSTEM_DESIGN: {
            "system_role": (
                "You are an expert system architect. REMEMBER generate the mermaid code by FOCUSING ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT abstract away real technical components — name actual services, databases, queues, and APIs from the project."
            ),
            "requirements": (
                "- Use graph TD syntax\n"
                "- Show actual system components: API gateway, services, databases, caches, queues\n"
                "- Show data flow between components with labeled arrows\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN: Do NOT use generic labels like 'Backend', 'Storage', 'Business Logic' — use real technical component names\n"
                "- Example:\n"
                "graph TD\n"
                "    Client[Web Client] --> Gateway[API Gateway]\n"
                "    Gateway --> AuthSvc[Auth Service]\n"
                "    Gateway --> VideoSvc[Video Ingestion Service]\n"
                "    VideoSvc --> Kafka[Kafka Queue]\n"
                "    Kafka --> AIPipeline[AI Inference Pipeline]\n"
                "    AIPipeline --> Redis[Redis Cache]\n"
                "    AIPipeline --> PostgreSQL[PostgreSQL DB]\n"
                "    Redis --> Dashboard[Dashboard API]\n"
                "    Dashboard --> Client"
            ),
        },
        DiagramType.FLOWCHART: {
            "system_role": (
                "You are an expert technical process analyst. REMEMBER generate the mermaid code by FOCUSING ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT produce a vague business-phase-only diagram — show the actual technical process flow with real system steps and decision logic."
            ),
            "requirements": (
                "- Use graph TD syntax\n"
                "- Show the technical process flow with real system steps and decision branches\n"
                "- Use {} for decisions, [] for steps\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN: Do NOT use purely business-language labels like 'Stakeholder Approval', 'Go Live', 'Business Review'\n"
                "- Example:\n"
                "graph TD\n"
                "    Discovery[Discovery Phase] --> DesignReady{Design Ready?}\n"
                "    DesignReady -->|Yes| Design[Design Phase]\n"
                "    DesignReady -->|No| Discovery\n"
                "    Design --> DevReady{Dev Ready?}\n"
                "    DevReady -->|Yes| Development[Development Phase]\n"
                "    DevReady -->|No| Design\n"
                "    Development --> Testing[Testing Phase]\n"
                "    Testing --> PassQA{QA Passed?}\n"
                "    PassQA -->|Yes| Deployment[Deployment]\n"
                "    PassQA -->|No| Development"
            ),
        },
        DiagramType.ARCHITECTURE: {
            "system_role": (
                "You are an enterprise architect. REMEMBER generate the mermaid code by FOCUSING ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT abstract real infrastructure into vague business capability names — show actual technical architecture layers with real component names."
            ),
            "requirements": (
                "- Use graph TD syntax with subgraph to show architecture layers\n"
                "- Show infrastructure layers: Client, API, Services, Data, Infrastructure\n"
                "- Name real technical components (e.g. PostgreSQL, Redis, Kafka, API Gateway)\n"
                "- All node labels with spaces MUST use brackets: A[Label With Spaces]\n"
                "- Node IDs must be single words or camelCase\n"
                "- FORBIDDEN: Do NOT use only business capability labels like 'Customer Portal', 'Business Logic', 'Data Store' — include real technical names\n"
                "- Example:\n"
                "graph TD\n"
                "    subgraph ClientLayer[Client Layer]\n"
                "        Browser[Web Browser]\n"
                "        Mobile[Mobile App]\n"
                "    end\n"
                "    subgraph APILayer[API Layer]\n"
                "        Gateway[API Gateway]\n"
                "        Auth[Auth Service]\n"
                "    end\n"
                "    subgraph DataLayer[Data Layer]\n"
                "        DB[PostgreSQL]\n"
                "        Cache[Redis]\n"
                "        Queue[Kafka]\n"
                "    end\n"
                "    Browser --> Gateway\n"
                "    Mobile --> Gateway\n"
                "    Gateway --> Auth\n"
                "    Auth --> DB\n"
                "    Gateway --> Cache\n"
                "    Gateway --> Queue"
            ),
        },
        DiagramType.GANTT: {
            "system_role": (
                "You are an expert project manager creating a technical delivery timeline for engineers. "
                "REMEMBER generate the code by FOCUSING ON the input PROPOSAL and PROMPT GIVEN. "
                "FORBIDDEN: Do NOT use only business milestone names — include actual engineering tasks like backend development, API integration, testing, CI/CD setup, and deployment."
            ),
            "requirements": (
                "- You MUST start with 'gantt' on the first line — no other syntax\n"
                "- You MUST include 'title' and 'dateFormat YYYY-MM-DD'\n"
                "- Use 'section' to group tasks by phase\n"
                "- Each task: TaskName :id, startDate_or_after_dep, duration\n"
                "- Include engineering tasks: backend dev, frontend dev, API integration, testing, CI/CD, deployment\n"
                "- FORBIDDEN: Do NOT produce only high-level business milestones — engineers need task-level detail\n"
                "- Example:\n"
                "gantt\n"
                "    title Project Timeline\n"
                "    dateFormat YYYY-MM-DD\n"
                "    section Discovery\n"
                "    Requirements :a1, 2026-05-14, 5d\n"
                "    section Design\n"
                "    Architecture :a2, after a1, 7d\n"
                "    section Development\n"
                "    Backend :a3, after a2, 14d\n"
                "    Frontend :a4, after a2, 14d\n"
                "    section Testing\n"
                "    QA Testing :a5, after a3, 7d\n"
                "    section Deployment\n"
                "    Release :milestone, after a5, 0d"
            ),
        },
    }

    def __init__(self, session_manager: Optional[Any] = None) -> None:
        """Initialize image generator agent."""
        super().__init__("image_agent", session_manager=session_manager)

        self.llm: Optional[InferenceClient] = None

        self.logger.info("Initializing ImageGeneratorAgent.")

        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize Hugging Face LLM."""
        self.logger.info("Initializing image generation LLM.")

        if not Config.HF_TOKEN:
            self.logger.error("HF_TOKEN missing in configuration.")
            raise ValueError("HF_TOKEN must be set.")

        self.llm = InferenceClient(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=Config.HF_TOKEN,
        )

        self.logger.info(
            "Image generator LLM initialized successfully with model=%s",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        )

    def _extract_text(self, response: Any) -> str:
        """Safely extract text from LLM response."""
        self.logger.info(
            "Extracting text from response type=%s",
            type(response).__name__,
        )

        if response is None:
            self.logger.warning("Received empty response object.")
            return ""

        if isinstance(response, str):
            self.logger.info(
                "Response is raw string with length=%d",
                len(response),
            )
            return response

        extracted = getattr(response, "content", str(response))

        self.logger.info(
            "Extracted response text length=%d",
            len(extracted),
        )

        return extracted

    def _build_diagram_prompt(
        self,
        plan: str,
        diagram_type: DiagramType,
        timeline: Optional[str] = None,
        audience_type: str = "engineer",
    ) -> str:
        """Build prompt for diagram generation."""
        self.logger.info(
            "Building prompt for diagram_type=%s audience=%s",
            diagram_type.value,
            audience_type,
        )

        self.logger.info(
            "Plan length=%d | Timeline length=%d",
            len(plan),
            len(timeline) if timeline else 0,
        )

        if audience_type == "stakeholder":
            template_config = self.STAKEHOLDER_DIAGRAM_TEMPLATES.get(
                diagram_type,
                self.STAKEHOLDER_DIAGRAM_TEMPLATES[DiagramType.FLOWCHART],
            )
        else:
            template_config = self.DIAGRAM_TEMPLATES.get(
                diagram_type,
                self.DIAGRAM_TEMPLATES[DiagramType.FLOWCHART],
            )

        prompt_template = PromptTemplate(
    input_variables=[
        "system_role",
        "rules",
        "plan",
        "timeline",
        "diagram_type_name",
    ],
    template="""
{system_role}

STRICT RULES — follow exactly:
{rules}

Use the plan below to fill in real project-specific content (services, phases, tasks).
Do NOT copy the example verbatim — adapt it to the actual project.

Plan:
{plan}

Timeline Reference:
{timeline}

Output ONLY the Mermaid diagram. No markdown fences. No explanation. No text after the last line.
""",
)

        formatted_prompt = prompt_template.format(
            system_role=template_config["system_role"],
            rules=template_config["requirements"],
            plan=plan,
            timeline=timeline or "No timeline provided",
            diagram_type_name=diagram_type.value,
        )

        self.logger.info(
            "Prompt built successfully | length=%d",
            len(formatted_prompt),
        )

        self.logger.info(
            "Prompt preview:\n%s",
            formatted_prompt[:1000],
        )

        return formatted_prompt

    def _parse_diagram(self, response: str) -> str:
        """Extract Mermaid diagram from response."""
        self.logger.info("Parsing Mermaid diagram from LLM response.")

        diagram = self._extract_text(response).strip()

        self.logger.info("Raw diagram length=%d", len(diagram))

        # Strip DeepSeek-R1 <think>...</think> reasoning blocks
        diagram = re.sub(r"<think>.*?</think>", "", diagram, flags=re.DOTALL).strip()

        # Extract from ```mermaid ... ``` fences if present
        if "```" in diagram:
            self.logger.warning("Markdown fences detected. Cleaning response.")
            # Try to extract just the content inside the first ```mermaid...``` block
            mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", diagram, re.DOTALL | re.IGNORECASE)
            if mermaid_match:
                diagram = mermaid_match.group(1).strip()
            else:
                diagram = (
                    diagram.replace("```mermaid", "")
                    .replace("```", "")
                    .strip()
                )

        # If there's preamble before the diagram keyword, strip it
        valid_starts = ["gantt", "flowchart", "graph", "sequenceDiagram",
                        "classDiagram", "stateDiagram", "erDiagram", "pie", "journey", "gitGraph"]
        lines = diagram.split("\n")
        for i, line in enumerate(lines):
            if any(line.strip().lower().startswith(v.lower()) for v in valid_starts):
                diagram = "\n".join(lines[i:]).strip()
                break

        # Strip trailing prose that LLMs append after the diagram
        diagram = self._strip_trailing_prose(diagram)

        # Sanitize unquoted multi-word node IDs (e.g. "Discovery --> Requirement Analysis" → "Discovery --> RequirementAnalysis")
        diagram = self._sanitize_node_ids(diagram)

        self.logger.info("Parsed diagram length=%d", len(diagram))
        self.logger.info("Diagram preview:\n%s", diagram[:1000])

        return diagram

    def _sanitize_node_ids(self, diagram: str) -> str:
        """Fix common LLM-generated Mermaid syntax errors.
        
        - Removes parentheses inside [] node labels (breaks Mermaid parser)
        - Collapses unquoted multi-word node IDs after arrows
        """
        lines = diagram.split("\n")
        sanitized = []
        
        for line in lines:
            # Remove parentheses inside [] node labels: A[Label (detail)] → A[Label detail]
            line = re.sub(r'\[([^\]]*)\(([^\)]*)\)([^\]]*)\]', r'[\1\2\3]', line)
            
            # Skip further processing if line uses brackets/braces (labels already safe)
            if "[" in line or "{" in line:
                sanitized.append(line)
                continue
            
            # Skip lines with () node labels (stadium shape) — those are intentional
            if "(" in line:
                sanitized.append(line)
                continue
            
            # Fix unquoted multi-word node IDs after arrows
            line = re.sub(
                r'(-->|->)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                lambda m: f"{m.group(1)} {m.group(2).replace(' ', '')}",
                line
            )
            line = re.sub(
                r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(-->|->)',
                lambda m: f"{m.group(1).replace(' ', '')} {m.group(2)}",
                line
            )
            
            sanitized.append(line)
        
        return "\n".join(sanitized)

    def _strip_trailing_prose(self, diagram: str) -> str:
        """Remove explanatory prose appended after the Mermaid diagram."""
        lines = diagram.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped and any(
                stripped.startswith(prefix)
                for prefix in ["This ", "The ", "Note:", "Here ", "Above ", "Below "]
            ):
                self.logger.info("Stripping trailing prose: %s", stripped[:60])
                break
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _build_fallback_gantt(self, timeline: Optional[str] = None) -> str:
        """Build a minimal valid gantt diagram when the LLM returns wrong syntax."""
        import datetime
        start = datetime.date.today().strftime("%Y-%m-%d")
        return (
            f"gantt\n"
            f"    title Project Timeline\n"
            f"    dateFormat YYYY-MM-DD\n"
            f"    section Discovery\n"
            f"    Requirements Analysis :a1, {start}, 5d\n"
            f"    section Design\n"
            f"    System Design :a2, after a1, 7d\n"
            f"    section Development\n"
            f"    Backend Development :a3, after a2, 14d\n"
            f"    Frontend Development :a4, after a2, 14d\n"
            f"    section Testing\n"
            f"    QA Testing :a5, after a3, 7d\n"
            f"    section Deployment\n"
            f"    Release :milestone, after a5, 0d"
        )

    def _validate_diagram(self, diagram: str) -> bool:
        """Validate Mermaid syntax structure."""
        self.logger.info("Validating generated Mermaid diagram.")

        if not diagram or len(diagram.strip()) < 10:
            self.logger.warning(
                "Diagram too short or empty."
            )
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

        self.logger.info(
            "Diagram starts with: %s",
            diagram_lower[:50],
        )

        is_valid = any(
            diagram_lower.startswith(v.lower())
            for v in valid_starts
        )

        self.logger.info(
            "Diagram validation result=%s",
            is_valid,
        )

        return is_valid

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Check whether error is retryable."""
        self.logger.warning(
            "Checking retryable status for error=%s",
            exc,
        )

        status_code = None

        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)

            if response:
                status_code = getattr(response, "status_code", None)

            if status_code is None:
                status_code = getattr(exc, "status_code", None)

            if status_code is None and exc.args:
                match = re.search(r"(\d{3})\s", str(exc.args[0]))
                if match:
                    status_code = int(match.group(1))

            self.logger.warning(
                "HF HTTP status code=%s",
                status_code,
            )

            return status_code in (
                408,
                429,
                500,
                502,
                503,
                504,
            )

        if isinstance(exc, requests.exceptions.HTTPError):
            response = getattr(exc, "response", None)
            if response:
                status_code = getattr(response, "status_code", None)
            self.logger.warning(
                "Requests HTTP status code=%s",
                status_code,
            )
            return status_code in (
                408,
                429,
                500,
                502,
                503,
                504,
            )

        retryable = isinstance(
            exc,
            (TimeoutError, ConnectionError),
        )

        self.logger.warning(
            "Retryable generic exception=%s",
            retryable,
        )

        return retryable

    def _chat_completion_with_retry(
        self,
        messages: list,
        max_tokens: int,
    ) -> Any:
        """Execute LLM request with retries."""
        last_exception = None

        max_attempts = 3
        base_delay = 1.0

        self.logger.info(
            "Starting diagram LLM request | max_tokens=%d",
            max_tokens,
        )

        self.logger.info(
            "Message preview:\n%s",
            messages[0]["content"][:1000],
        )

        for attempt in range(max_attempts):
            try:
                self.logger.info(
                    "LLM attempt %d/%d",
                    attempt + 1,
                    max_attempts,
                )

                start_time = time.time()

                response = self.llm.chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                )

                elapsed = round(time.time() - start_time, 2)

                self.logger.info(
                    "LLM request succeeded in %.2fs",
                    elapsed,
                )

                if hasattr(response, "choices"):
                    content = response.choices[0].message.content

                    self.logger.info(
                        "Response content length=%d",
                        len(content) if content else 0,
                    )

                    self.logger.info(
                        "Response preview:\n%s",
                        content[:1000] if content else "EMPTY",
                    )

                return response

            except Exception as exc:
                last_exception = exc

                self.logger.error(
                    "LLM request failed on attempt %d/%d | error=%s",
                    attempt + 1,
                    max_attempts,
                    exc,
                    exc_info=True,
                )

                if not self._is_retryable_error(exc):
                    self.logger.error(
                        "Encountered non-retryable error."
                    )
                    raise exc

                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)

                    self.logger.warning(
                        "Retrying in %.1f seconds.",
                        delay,
                    )

                    time.sleep(delay)

        self.logger.error(
            "All LLM attempts failed."
        )

        raise last_exception

    def generate_diagram(
        self,
        plan: str,
        diagram_type: DiagramType,
        timeline: Optional[str] = None,
        audience_type: str = "engineer",
    ) -> dict[str, Any]:
        """Generate a Mermaid diagram."""
        try:
            self.logger.info(
                "Generating diagram type=%s",
                diagram_type.value,
            )

            if not plan:
                self.logger.error(
                    "Plan missing. Cannot generate diagram."
                )

                return {
                    "diagram_type": diagram_type.value,
                    "mermaid_code": None,
                    "is_valid": False,
                    "error": "No plan provided",
                    "image_data": None,
                }

            self.logger.info(
                "Plan length=%d",
                len(plan),
            )

            prompt = self._build_diagram_prompt(
                plan,
                diagram_type,
                timeline,
                audience_type=audience_type,
            )

            response = self._chat_completion_with_retry(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=700,
            )

            msg = response.choices[0].message if hasattr(response, "choices") else None
            raw_content = (
                (msg and msg.content and msg.content.strip())
                or (getattr(msg, "reasoning_content", None) or "").strip()
                or (getattr(msg, "reasoning", None) or "").strip()
                or ""
            ) if msg else ""

            self.logger.info(
                "Raw LLM content length=%d",
                len(raw_content) if raw_content else 0,
            )

            diagram = self._parse_diagram(raw_content)

            # If gantt was requested but LLM returned wrong type, build a minimal valid gantt
            if diagram_type == DiagramType.GANTT and not diagram.lower().strip().startswith("gantt"):
                self.logger.warning(
                    "Gantt diagram requested but LLM returned wrong type. Using fallback gantt."
                )
                diagram = self._build_fallback_gantt(timeline)

            is_valid = self._validate_diagram(diagram)

            if not is_valid:
                self.logger.warning(
                    "Generated Mermaid diagram failed validation."
                )

                self.logger.warning(
                    "Invalid Mermaid output:\n%s",
                    diagram,
                )

                return {
                    "diagram_type": diagram_type.value,
                    "mermaid_code": diagram,
                    "is_valid": False,
                    "error": "No valid Mermaid diagram generated",
                    "image_data": None,
                }

            self.logger.info(
                "Diagram validated successfully."
            )

            image_data = None

            if MERMAID_AVAILABLE:
                max_render_attempts = 3
                render_delay = 1.0
                for render_attempt in range(max_render_attempts):
                    try:
                        self.logger.info(
                            "Attempting Mermaid PNG rendering (attempt %d/%d).",
                            render_attempt + 1,
                            max_render_attempts,
                        )
                        # Mermaid() constructor calls _make_request_to_mermaid()
                        # which sets img_response; raises MermaidError on failure
                        mm = Mermaid(diagram)
                        image_bytes = mm.img_response.content
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        image_data = f"data:image/png;base64,{image_base64}"
                        self.logger.info("Diagram image rendered successfully.")
                        break

                    except Exception as exc:
                        self.logger.warning(
                            "Failed Mermaid rendering (attempt %d/%d) | error=%s",
                            render_attempt + 1,
                            max_render_attempts,
                            exc,
                        )
                        if render_attempt < max_render_attempts - 1:
                            time.sleep(render_delay)
                            render_delay *= 2

            else:
                self.logger.warning("Mermaid renderer unavailable.")

            return {
                "diagram_type": diagram_type.value,
                "mermaid_code": diagram,
                "title": diagram_type.value.replace("_", " ").title(),
                "description": (
                    f"Auto-generated {diagram_type.value} diagram"
                ),
                "is_valid": True,
                "validation_feedback": "Diagram generated successfully",
                "error": None,
                "image_data": image_data,
            }

        except Exception as exc:
            self.logger.error(
                "Diagram generation failed for type=%s | error=%s",
                diagram_type.value,
                exc,
                exc_info=True,
            )

            return {
                "diagram_type": diagram_type.value,
                "mermaid_code": None,
                "is_valid": False,
                "error": (
                    f"Failed to generate {diagram_type.value} "
                    f"diagram: {exc}"
                ),
                "image_data": None,
            }

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate all requested diagrams."""
        try:
            self.logger.info(
                "Starting image generation workflow."
            )

            plan = state.get("plan")
            timeline = state.get("timetable")
            audience_type = state.get("audience_type", "engineer")

            improvement_prompt = state.get("improvement_prompt")
            if improvement_prompt:
                self.logger.info("Using improvement prompt for regeneration")
                plan = f"{plan}\n\n--- IMPROVEMENT FEEDBACK ---\n{improvement_prompt}"

            diagram_types_raw = state.get(
                "diagram_types",
                [DiagramType.WORKFLOW],
            )

            self.logger.info(
                "Plan exists=%s | timeline exists=%s",
                bool(plan),
                bool(timeline),
            )

            self.logger.info(
                "Requested diagram types=%s",
                diagram_types_raw,
            )

            if not plan:
                self.logger.error(
                    "No plan provided to image generator."
                )

                return self._update_state(
                    state,
                    {
                        "error": (
                            "Failed to generate images: "
                            "no plan provided"
                        ),
                        "diagrams": [],
                        "current_agent": "image_agent",
                    },
                )

            # Parse diagram types
            if isinstance(diagram_types_raw, list):
                if all(
                    isinstance(d, str)
                    for d in diagram_types_raw
                ):
                    diagram_types = [
                        DiagramType(d)
                        for d in diagram_types_raw
                    ]
                else:
                    diagram_types = diagram_types_raw

            elif isinstance(diagram_types_raw, str):
                diagram_types = [
                    DiagramType(diagram_types_raw)
                ]

            else:
                diagram_types = [DiagramType.WORKFLOW]

            self.logger.info(
                "Resolved %d diagram types.",
                len(diagram_types),
            )

            # Stakeholders only receive business-friendly diagram types
            if audience_type == "stakeholder":
                diagram_types = [
                    dt for dt in diagram_types
                    if dt in self.STAKEHOLDER_DIAGRAM_TYPES
                ]
                if not diagram_types:
                    diagram_types = [DiagramType.FLOWCHART, DiagramType.GANTT]
                self.logger.info(
                    "Stakeholder audience: filtered to %d diagram types.",
                    len(diagram_types),
                )

            diagrams = []

            previous_invalid = self._get_previous_invalid_output("diagrams")
            if previous_invalid:
                self.logger.info(
                    "Found previous invalid output for reference"
                )

            for idx, diagram_type in enumerate(diagram_types):
                self.logger.info(
                    "Generating diagram %d/%d | type=%s",
                    idx + 1,
                    len(diagram_types),
                    diagram_type.value,
                )

                diagram_result = self.generate_diagram(
                    plan=plan,
                    diagram_type=diagram_type,
                    timeline=timeline,
                    audience_type=audience_type,
                )

                self.logger.info(
                    "Diagram generation result | valid=%s | error=%s",
                    diagram_result.get("is_valid"),
                    diagram_result.get("error"),
                )

                diagrams.append(diagram_result)

            valid_count = sum(
                1 for d in diagrams if d.get("is_valid")
            )

            self.logger.info(
                "Generated %d/%d valid diagrams.",
                valid_count,
                len(diagrams),
            )

            for idx, diagram in enumerate(diagrams):
                self.logger.info(
                    "Diagram #%d summary | type=%s | valid=%s",
                    idx + 1,
                    diagram.get("diagram_type"),
                    diagram.get("is_valid"),
                )

            result_state = self._update_state(
                state,
                {
                    "diagrams": diagrams,
                    "diagram_count": len(diagrams),
                    "valid_diagram_count": valid_count,
                    "error": (
                        None
                        if valid_count > 0
                        else "No valid diagrams generated"
                    ),
                    "current_agent": "image_agent",
                },
            )

            self._save_session_output(
                output_type="diagrams",
                output_data={
                    "diagrams": diagrams,
                    "diagram_count": len(diagrams),
                    "valid_diagram_count": valid_count,
                },
                feedback=result_state.get("error"),
                is_valid=valid_count > 0,
            )

            return result_state

        except Exception as exc:
            self.logger.error(
                "Image generator workflow failed | error=%s",
                exc,
                exc_info=True,
            )

            return self._update_state(
                state,
                {
                    "error": (
                        f"Image generator agent failed: {exc}"
                    ),
                    "diagrams": [],
                    "diagram_count": 0,
                    "valid_diagram_count": 0,
                    "current_agent": "image_agent",
                },
            )