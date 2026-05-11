"""Prompt optimization module using LangChain for FlowForge."""
from typing import Any

from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from src.logger import setup_logger
from src.config import Config

logger = setup_logger()


class PromptOptimizer:
    """Optimize user prompts for better agent performance using LangChain."""

    def __init__(self) -> None:
        """Initialize the prompt optimizer with LLM router chains."""
        logger.debug("Initializing PromptOptimizer...")
        self.llm = self._initialize_llm()
        self._optimizer_chain = self._build_optimizer_chain()
        self._proposal_extractor_chain = self._build_proposal_extractor()
        self._prompt_enhancer_chain = self._build_prompt_enhancer()
        logger.info("PromptOptimizer initialized successfully")

    def _initialize_llm(self) -> HuggingFaceEndpoint:
        """Initialize HuggingFace LLM for prompt optimization."""
        logger.debug("Initializing HuggingFace LLM endpoint...")
        try:
            llm = HuggingFaceEndpoint(
                repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                task="text-generation",
                max_new_tokens=512,
                temperature=0.5,
                huggingfacehub_api_token=Config.HF_TOKEN,
            )
            logger.debug("HuggingFace LLM endpoint initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

    def _build_optimizer_chain(self) -> RunnableSequence:
        """Build the main prompt optimization chain."""
        optimizer_template = """You are an expert prompt engineer. Optimize the following user prompt
for use with an AI agent pipeline that generates project documentation and diagrams.

The pipeline has these stages:
1. Timeline Agent - generates milestones, parallel work streams, and Gantt charts
2. Plan Agent - creates detailed project plans with resource allocation and risk analysis
3. Image Generator Agent - produces Mermaid diagrams (workflow, CI/CD, system design, flowcharts, architecture)
4. Validator Agent - validates diagram syntax and logical consistency

Optimization rules:
- Make requirements explicit and numbered
- Add implicit constraints that experts would consider
- Structure the prompt for each pipeline stage
- Include diagram-specific requirements
- Ensure technical accuracy and completeness
- Remove ambiguity

User Prompt:
{user_prompt}

Optimized Prompt:"""

        prompt_template = PromptTemplate(
            input_variables=["user_prompt"],
            template=optimizer_template,
        )
        return prompt_template | self.llm

    def _build_proposal_extractor(self) -> RunnableSequence:
        """Build chain to extract structured proposal from raw text."""
        extractor_template = """You are an expert at extracting structured project proposals from unstructured text.

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

        prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=extractor_template,
        )
        return prompt_template | self.llm

    def _build_prompt_enhancer(self) -> RunnableSequence:
        """Build chain to enhance prompts per diagram type."""
        enhancer_template = """You are a specialized prompt enhancer for diagram generation.

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

        prompt_template = PromptTemplate(
            input_variables=["diagram_type", "plan", "user_prompt"],
            template=enhancer_template,
        )
        return prompt_template | self.llm

    def optimize(self, user_prompt: str) -> dict[str, Any]:
        """
        Optimize a user prompt for the FlowForge pipeline.

        Args:
            user_prompt: The raw user prompt to optimize.

        Returns:
            Dict containing original prompt, optimized prompt, and metadata.
        """
        logger.info(f"Starting prompt optimization for input: {user_prompt[:50]}...")
        try:
            optimized = self._optimizer_chain.invoke({"user_prompt": user_prompt})
            optimized_text = (
                optimized.content if hasattr(optimized, "content") else str(optimized)
            )
            logger.info("Prompt optimization completed successfully")
            return {
                "original_prompt": user_prompt,
                "optimized_prompt": optimized_text.strip(),
                "optimization_technique": "langchain_llm_router",
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
        try:
            result = self._proposal_extractor_chain.invoke({"user_input": raw_input})
            if isinstance(result, str):
                result_text = result
            else:
                content = getattr(result, "content", None)

            if isinstance(content, str):
                result_text = content
            else:
                result_text = str(result)

            # Parse the JSON from the response
            import re
            import json

            json_match = re.search(r"```json\s*\n(.*?)\n```", result_text, re.DOTALL)
            if json_match:
                try:
                    proposal = json.loads(json_match.group(1))
                    logger.debug(f"Successfully extracted proposal: {proposal.get('title', 'Unknown')}")
                    return proposal
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from extractor response: {str(e)}")

            # Fallback: return raw text structured minimally
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
            result = self._prompt_enhancer_chain.invoke(
                {
                    "diagram_type": diagram_type,
                    "plan": plan,
                    "user_prompt": user_prompt,
                }
            )

            if isinstance(result, str):
                result_text = result
            else:
                content = getattr(result, "content", None)

            if isinstance(content, str):
                result_text = content
            else:
                result_text = str(result)

            logger.debug(f"Successfully enhanced prompt for {diagram_type} diagram")
            return result_text.strip()
        except Exception as e:
            logger.error(f"Prompt enhancement failed for diagram type '{diagram_type}': {str(e)}")
            raise