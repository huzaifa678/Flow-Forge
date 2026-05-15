"""Base Validator Agent for FlowForge - abstract base for all validators."""

import re
import time
from abc import abstractmethod
from typing import Any, Optional

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from src.agents.base_agent import BaseAgent
from src.config import Config


class BaseValidatorAgent(BaseAgent):
    """Abstract base class for all FlowForge validator agents.

    Provides common validation infrastructure including:
    - LLM initialization with retry-capable chat completion
    - Basic Mermaid syntax validation
    - LLM response parsing for structured validation results
    - Subclass-specific routing decision support
    """

    VALIDATOR_SYSTEM_PROMPT = """
You are a Mermaid.js syntax validator. Validate only real rendering errors.

VALID Mermaid patterns you must NOT flag as errors:
- `graph TD;` or `graph LR;` — trailing semicolons on the root declaration are valid
- A node having multiple outbound edges (e.g. `A --> B`, `A --> C`) — this is valid
- A node having multiple inbound edges — this is valid
- Gantt sections with tasks — sections do not need explicit closing
- `graph TD` and `flowchart TD` are both valid root declarations
- Node IDs with numbers like `a1`, `b1`, `cc1` are valid
- Using both `[]`, `{}`, and `()` node shapes in the same `graph TD` is valid — this is NOT mixed types
- Standalone node declarations like `A[Label]` without edges are valid
- Duplicate edges between the same nodes are valid (unusual but not a syntax error)

Only mark INVALID for these real errors:
1. Diagram does not start with a valid keyword (graph, flowchart, gantt, sequenceDiagram, etc.)
2. Mismatched brackets `[]`, `()`, `{}` that would break parsing
3. Incomplete arrow syntax (e.g. `A -` with nothing after)
4. Text or prose appearing outside node/edge definitions
5. Unclosed subgraph blocks (subgraph without end)
6. Parentheses inside square bracket labels: `A[Label (detail)]` — this breaks the parser

Do NOT mark INVALID based on:
- Semicolons on declarations
- Number of edges per node
- "Mixed diagram types" — using different node shapes in graph TD is NOT mixed types
- Indentation style

Return ONLY this exact format:
VALID: true/false
CRITICAL_ISSUES: [list or None]
WARNINGS: [list or None]
SUGGESTIONS: [list or None]
FEEDBACK: [one sentence explanation]
"""

    def __init__(self, name: str, session_manager: Optional[Any] = None) -> None:
        """
        Initialize base validator agent.

        Parameters
        ----------
        name : str
            Name of the validator agent.
        session_manager : SessionManager, optional
            Session manager for database persistence.
        """
        super().__init__(name, session_manager=session_manager)

        self.llm: InferenceClient | None = None
        self._initialize_llm()

    @abstractmethod
    def _get_model(self) -> str:
        """Return the HuggingFace model identifier for this validator."""

    def _initialize_llm(self) -> None:
        """Initialize Hugging Face LLM."""
        if not Config.HF_TOKEN:
            self.logger.error("HF_TOKEN missing in configuration.")
            raise ValueError("HF_TOKEN must be set.")

        self.llm = InferenceClient(
            model=self._get_model(),
            token=Config.HF_TOKEN
        )

        self.logger.info(
            "%s initialized successfully with model=%s",
            self.name,
            self._get_model(),
        )

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Check whether exception is retryable."""
        if isinstance(exc, HfHubHTTPError):
            response = getattr(exc, "response", None)

            if response:
                status_code = getattr(response, "status_code", None)

                self.logger.warning(
                    "HF Hub HTTP error status=%s",
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

        retryable = isinstance(exc, (TimeoutError, ConnectionError))

        self.logger.warning(
            "Retryable generic error=%s | retryable=%s",
            type(exc).__name__,
            retryable,
        )

        return retryable

    def _chat_completion_with_retry(
        self,
        messages: list,
        max_tokens: int,
    ) -> Any:
        """Execute chat completion with retry logic."""
        last_exception = None

        max_attempts = 3
        base_delay = 1.0

        self.logger.info(
            "Starting LLM request | max_tokens=%d",
            max_tokens,
        )

        self.logger.info(
            "Prompt preview:\n%s",
            messages[0]["content"][:500],
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
                    "LLM success on attempt %d | duration=%.2fs",
                    attempt + 1,
                    elapsed,
                )

                if hasattr(response, "choices"):
                    content = response.choices[0].message.content

                    self.logger.info(
                        "Response length=%d",
                        len(content) if content else 0,
                    )

                    self.logger.info(
                        "Response preview:\n%s",
                        content[:500] if content else "EMPTY",
                    )

                return response

            except Exception as exc:
                last_exception = exc

                self.logger.error(
                    "LLM call failed on attempt %d/%d | error=%s",
                    attempt + 1,
                    max_attempts,
                    exc,
                    exc_info=True,
                )

                if not self._is_retryable_error(exc):
                    self.logger.error(
                        "Non-retryable error encountered."
                    )
                    raise exc

                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)

                    self.logger.warning(
                        "Retrying LLM in %.1f seconds...",
                        delay,
                    )

                    time.sleep(delay)

        self.logger.error(
            "LLM failed after %d attempts.",
            max_attempts,
        )

        raise last_exception

    def _basic_mermaid_validation(
        self,
        diagram: str,
    ) -> dict[str, Any]:
        """Perform basic Mermaid syntax validation.

        Parameters
        ----------
        diagram : str
            Mermaid diagram code to validate (may include markdown fences).

        Returns
        -------
        dict[str, Any]
            Validation result with is_valid and feedback keys.
        """
        self.logger.info("Running basic Mermaid validation.")

        if not diagram or not diagram.strip():
            self.logger.warning("Diagram is empty.")
            return {
                "is_valid": False,
                "feedback": "Empty diagram.",
            }

        # Extract Mermaid code block from within ```mermaid ... ``` fences if present
        diagram_extracted = self._extract_mermaid_code_block(diagram)
        self.logger.info(
            "Extracted diagram (first 100 chars): %s",
            diagram_extracted[:100],
        )

        valid_starts = [
            "gantt",
            "flowchart",
            "graph",
            "sequenceDiagram",
            "classDiagram",
            "stateDiagram",
            "erDiagram",
            "pie",
            "journey",
            "gitGraph",
        ]

        diagram_lower = diagram_extracted.lower().strip()

        self.logger.info(
            "Diagram starts with: %s",
            diagram_lower[:50],
        )

        if not any(
            diagram_lower.startswith(v.lower())
            for v in valid_starts
        ):
            self.logger.warning(
                "Invalid Mermaid root syntax detected."
            )
            return {
                "is_valid": False,
                "feedback": (
                    "Diagram does not start with valid Mermaid syntax. "
                    f"Got: '{diagram_lower[:50]}...'"
                ),
            }

        for open_char, close_char in [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
        ]:
            open_count = diagram_extracted.count(open_char)
            close_count = diagram_extracted.count(close_char)

            self.logger.info(
                "Bracket check %s/%s | open=%d close=%d",
                open_char,
                close_char,
                open_count,
                close_count,
            )

            if open_count != close_count:
                self.logger.warning(
                    "Bracket mismatch detected for %s/%s",
                    open_char,
                    close_char,
                )
                return {
                    "is_valid": False,
                    "feedback": (
                        f"Mismatched '{open_char}'/'{close_char}' brackets."
                    ),
                }

        self.logger.info("Basic Mermaid validation passed.")

        # Check for parentheses inside square bracket labels — breaks Mermaid parser
        if re.search(r'\[[^\]]*\([^\)]*\)[^\]]*\]', diagram_extracted):
            self.logger.warning("Parentheses inside [] node label detected.")
            return {
                "is_valid": False,
                "feedback": "Parentheses inside [] node labels break the Mermaid parser.",
            }

        return {
            "is_valid": True,
            "feedback": "Validation passed.",
        }

    def _extract_mermaid_code_block(self, text: str) -> str:
        """Extract Mermaid diagram code from fenced code blocks.

        Parameters
        ----------
        text : str
            Text that may contain ```mermaid ... ``` code blocks.

        Returns
        -------
        str
            The extracted Mermaid diagram code, or the original text if no
            fenced code block is found.
        """
        import re

        # Strip DeepSeek-R1 <think>...</think> reasoning blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Try to find ```mermaid ... ``` block
        mermaid_pattern = r"```mermaid\s*(.*?)\s*```"
        match = re.search(mermaid_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            self.logger.info("Extracted Mermaid code from fenced block.")
            return self._strip_trailing_prose(extracted)

        # Try to find ```...``` block (any language) that looks like mermaid
        generic_pattern = r"```(.*?)```"
        matches = re.findall(generic_pattern, text, re.DOTALL)
        for block in matches:
            block_lower = block.lower().strip()
            if any(
                block_lower.startswith(keyword)
                for keyword in ["gantt", "flowchart", "graph", "sequence", "class", "state", "er", "pie", "journey", "gitgraph"]
            ):
                self.logger.info("Extracted Mermaid code from generic fenced block.")
                return self._strip_trailing_prose(block.strip())

        # No fenced code block found; try to find the diagram start in raw text
        valid_starts = ["gantt", "flowchart", "graph", "sequenceDiagram",
                        "classDiagram", "stateDiagram", "erDiagram", "pie", "journey", "gitGraph"]
        lines = text.strip().split("\n")
        for i, line in enumerate(lines):
            if any(line.strip().lower().startswith(v.lower()) for v in valid_starts):
                extracted = "\n".join(lines[i:]).strip()
                self.logger.info("Extracted Mermaid code by finding diagram keyword in raw text.")
                return self._strip_trailing_prose(extracted)

        self.logger.info("No Mermaid code fences found, using raw text.")
        return self._strip_trailing_prose(text.strip())

    def _strip_trailing_prose(self, diagram: str) -> str:
        """Remove explanatory prose that LLMs append after Mermaid diagrams.

        Parameters
        ----------
        diagram : str
            Mermaid diagram code that may have trailing prose.

        Returns
        -------
        str
            Cleaned diagram with trailing prose removed.
        """
        lines = diagram.split("\n")
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Stop at lines that look like prose (start with "This", "The", "Note:", etc.)
            if stripped and any(
                stripped.startswith(prefix)
                for prefix in ["This ", "The ", "Note:", "Here ", "Above ", "Below "]
            ):
                self.logger.info("Stripping trailing prose starting with: %s", stripped[:50])
                break
            
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines).strip()

    def _parse_llm_validation(
        self,
        response: str,
    ) -> dict[str, Any]:
        """Parse validator LLM response.

        Parameters
        ----------
        response : str
            Raw LLM response text.

        Returns
        -------
        dict[str, Any]
            Parsed validation result with is_valid, feedback, critical_issues,
            warnings, and suggestions keys.
        """
        self.logger.info("Parsing validator LLM response.")

        result = {
            "is_valid": True,
            "feedback": "",
            "critical_issues": [],
            "warnings": [],
            "suggestions": [],
        }

        valid_match = re.search(
            r"VALID:\s*(true|false)",
            response,
            re.I,
        )

        if valid_match:
            result["is_valid"] = (
                valid_match.group(1).lower() == "true"
            )

        critical_match = re.search(
            r"CRITICAL_ISSUES:\s*(.+?)(?:\n(?:WARNINGS|SUGGESTIONS|FEEDBACK|$))",
            response,
            re.I | re.DOTALL,
        )

        if critical_match:
            issues = critical_match.group(1).strip()

            if issues.lower() != "none":
                result["critical_issues"] = [
                    i.strip()
                    for i in re.split(r"[\n,]", issues)
                    if i.strip() and i.strip() != "-"
                ]

        warnings_match = re.search(
            r"WARNINGS:\s*(.+?)(?:\n(?:SUGGESTIONS|FEEDBACK|$))",
            response,
            re.I | re.DOTALL,
        )

        if warnings_match:
            warnings = warnings_match.group(1).strip()

            if warnings.lower() != "none":
                result["warnings"] = [
                    i.strip()
                    for i in re.split(r"[\n,]", warnings)
                    if i.strip() and i.strip() != "-"
                ]

        suggestions_match = re.search(
            r"SUGGESTIONS:\s*(.+?)(?:\n(?:FEEDBACK|$))",
            response,
            re.I | re.DOTALL,
        )

        if suggestions_match:
            suggestions = suggestions_match.group(1).strip()

            if suggestions.lower() != "none":
                result["suggestions"] = [
                    i.strip()
                    for i in re.split(r"[\n,]", suggestions)
                    if i.strip() and i.strip() != "-"
                ]

        feedback_match = re.search(
            r"FEEDBACK:\s*(.*)",
            response,
            re.I | re.S,
        )

        if feedback_match:
            result["feedback"] = feedback_match.group(1).strip()

        self.logger.info(
            "Parsed validation summary | valid=%s | critical=%d | warnings=%d | suggestions=%d",
            result["is_valid"],
            len(result["critical_issues"]),
            len(result["warnings"]),
            len(result["suggestions"]),
        )

        return result