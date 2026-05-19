"""Prompt optimization module using InferenceClient for FlowForge."""
from typing import Any

from huggingface_hub import InferenceClient

from src.logger import setup_logger
from src.config import Config

logger = setup_logger()


class PromptOptimizer:
    """Optimize user prompts for better agent performance using InferenceClient."""

    ENGINEER_OPTIMIZER_TEMPLATE = """You are an expert prompt engineer optimizing for a TECHNICAL engineering audience.

The pipeline has these stages:
1. Timeline Agent - generates milestones, parallel work streams, and Gantt charts
2. Plan Agent - creates detailed project plans with resource allocation and risk analysis
3. Image Generator Agent - produces Mermaid diagrams (workflow, CI/CD, system design, flowcharts, architecture)
4. Validator Agent - validates diagram syntax and logical consistency

Optimization rules:
- Make requirements explicit and numbered
- Add implicit technical constraints that engineers would consider (scalability, latency, fault tolerance, security)
- Structure the prompt for each pipeline stage
- Include diagram-specific technical requirements (services, APIs, databases, queues, infra)
- Ensure technical accuracy and completeness — name actual components and technologies
- Remove ambiguity
- FORBIDDEN: Do NOT simplify to high-level business phases only — engineers need real technical detail

User Prompt:
{user_prompt}

Optimized Prompt (engineering focus):"""

    STAKEHOLDER_OPTIMIZER_TEMPLATE = """You are an expert prompt engineer optimizing for a NON-TECHNICAL business stakeholder audience.

The pipeline will generate business-friendly diagrams and a condensed spec document for executives and decision-makers.

Optimization rules:
- Rewrite any technical language into plain business outcomes and value statements
- Focus on WHAT the system delivers, not HOW it is built internally
- Frame requirements as business capabilities and user-facing features
- Replace infrastructure/DevOps/ML details with timeline, cost, risk, and delivery milestones
- Keep the project goal, scope, team structure, and delivery timeline clear
- FORBIDDEN: Do NOT include CI/CD pipelines, system architecture specifics, tech stack names, infrastructure details, or engineering implementation details — stakeholders do not need these

User Prompt:
{user_prompt}

Optimized Prompt (stakeholder focus):"""

    EXTRACTOR_TEMPLATE = """You are an expert at extracting structured project proposals from unstructured text.

Extract the following fields from the user's input and return them in the exact JSON format:

```json
{{
    "title": "Project title (3-100 chars)",
    "description": "Detailed project description",
    "requirements": ["requirement 1", "requirement 2", ...],
    "constraints": ["constraint 1", "constraint 2", ...],
    "tech_stack": ["technology 1", "technology 2", ...],
    "timeline_weeks": <number>,
    "team_size": <number>,
    "budget_range": "$XXk-$YYk" or null
}}
```

User Input:
{user_input}

Structured Proposal:"""

    ENHANCER_TEMPLATE = """You are a specialized prompt enhancer for diagram generation.

Enhance the following prompt specifically for generating a {diagram_type} diagram.

The diagram must:
- Be valid Mermaid syntax
- Include all relevant components from the project plan
- Follow industry best practices for {diagram_type} diagrams
- Be production-ready and comprehensive

Project Plan:
{plan}

Original Prompt:
{user_prompt}

Enhanced Prompt for {diagram_type}:"""

    def __init__(self) -> None:
        """Initialize the prompt optimizer with LLM."""
        logger.debug("Initializing PromptOptimizer...")
        self.llm = self._initialize_llm()
        logger.info("PromptOptimizer initialized successfully")

    def _initialize_llm(self) -> InferenceClient:
        """Initialize HuggingFace LLM for prompt optimization."""
        logger.debug("Initializing HuggingFace LLM endpoint...")
        try:
            llm = InferenceClient(
                model="Qwen/Qwen2.5-Coder-32B-Instruct",
                token=Config.HF_TOKEN,
            )
            logger.debug("HuggingFace LLM endpoint initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _chat_complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Helper method to call chat_completion and extract text."""
        response = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return (
            response.choices[0].message.content
            if hasattr(response, "choices")
            else str(response)
        )

    def optimize(self, user_prompt: str, audience_type: str = "engineer") -> dict[str, Any]:
        """
        Optimize a user prompt for the FlowForge pipeline.

        Args:
            user_prompt: The raw user prompt to optimize.
            audience_type: "engineer" or "stakeholder" — selects the optimization strategy.

        Returns:
            Dict containing original prompt, optimized prompt, and metadata.
        """
        logger.info(
            "Starting prompt optimization | audience=%s | input=%s...",
            audience_type,
            user_prompt[:50],
        )
        try:
            template = (
                self.STAKEHOLDER_OPTIMIZER_TEMPLATE
                if audience_type == "stakeholder"
                else self.ENGINEER_OPTIMIZER_TEMPLATE
            )
            formatted_prompt = template.format(user_prompt=user_prompt)
            optimized_text = self._chat_complete(formatted_prompt, max_tokens=512)
            logger.info("Prompt optimization completed | audience=%s", audience_type)
            return {
                "original_prompt": user_prompt,
                "optimized_prompt": optimized_text.strip(),
                "optimization_technique": "inference_client",
                "audience_type": audience_type,
            }
        except Exception as e:
            logger.error(f"Prompt optimization failed: {str(e)}")
            raise

    def extract_proposal(self, raw_input: str) -> dict[str, Any]:
        """
        Extract structured proposal from raw user input.

        Args:
            raw_input: The raw user input text.

        Returns:
            Dict containing structured proposal fields.
        """
        logger.info(f"Starting proposal extraction for input: {raw_input[:50]}...")
        import json
        import re

        try:
            formatted_prompt = self.EXTRACTOR_TEMPLATE.format(user_input=raw_input)
            result_text = self._chat_complete(formatted_prompt, max_tokens=512)

            json_match = re.search(r"```json\s*\n(.*?)\n```", result_text, re.DOTALL)
            if json_match:
                try:
                    proposal = json.loads(json_match.group(1))
                    logger.debug(f"Successfully extracted proposal: {proposal.get('title', 'Unknown')}")
                    return proposal
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from extractor response: {str(e)}")

            logger.info("Using fallback proposal structure")
            return {
                "title": raw_input[:100],
                "description": raw_input,
                "requirements": [],
                "constraints": [],
                "tech_stack": [],
                "timeline_weeks": 12,
                "team_size": 5,
                "budget_range": None,
            }
        except Exception as e:
            logger.error(f"Proposal extraction failed: {str(e)}")
            raise

    def enhance_for_diagram(
        self,
        diagram_type: str,
        plan: str,
        user_prompt: str,
    ) -> str:
        """
        Enhance the prompt for a specific diagram type.

        Args:
            diagram_type: Type of diagram (workflow, ci_cd, system_design, etc.).
            plan: The project plan to base the diagram on.
            user_prompt: The original user prompt.

        Returns:
            Enhanced prompt string for the specific diagram type.
        """
        logger.info(f"Enhancing prompt for diagram type: {diagram_type}")
        try:
            formatted_prompt = self.ENHANCER_TEMPLATE.format(
                diagram_type=diagram_type,
                plan=plan,
                user_prompt=user_prompt,
            )
            result_text = self._chat_complete(formatted_prompt, max_tokens=512)
            logger.debug(f"Successfully enhanced prompt for {diagram_type} diagram")
            return result_text.strip()
        except Exception as e:
            logger.error(f"Prompt enhancement failed for diagram type '{diagram_type}': {str(e)}")
            raise