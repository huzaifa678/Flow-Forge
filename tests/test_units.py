"""Tests for FlowForge agents and workflow."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

from src.schemas.request import PromptRequest, AudienceType
from src.pipeline.prompt_optimizer import PromptOptimizer

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FlowForge'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.time_agent import TimeAgent
from src.agents.plan_agent import PlanAgent
from src.agents.image_generator_agent import ImageGeneratorAgent
from src.agents.validator_agent import ValidatorAgent
from src.agents.base_validator_agent import BaseValidatorAgent
from src.agents.time_validator_agent import TimeValidatorAgent
from src.workflow.graph_workflow import create_flowforge_workflow
from src.schemas.request import (
    DiagramType,
    AudienceType,
)


class TestBaseAgent(unittest.TestCase):
    """Test cases for base agent functionality."""

    def setUp(self):
        """Set up test fixtures."""
        pass


class TestPromptOptimizer(unittest.TestCase):
    """Test cases for PromptOptimizer module."""

    @patch('src.pipeline.prompt_optimizer.InferenceClient')
    def test_initialization(self, mock_llm):
        """Test PromptOptimizer initialization."""
        optimizer = PromptOptimizer()
        self.assertIsNotNone(optimizer.llm)
        mock_llm.assert_called_once()

    @patch('src.pipeline.prompt_optimizer.InferenceClient')
    def test_optimize_returns_structure(self, mock_llm):
        """Test optimize returns proper structure."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Optimized version of the prompt with more detail."))]
        )
        mock_llm.return_value = mock_instance

        optimizer = PromptOptimizer()
        result = optimizer.optimize("Create a web app")

        self.assertIn("original_prompt", result)
        self.assertIn("optimized_prompt", result)
        self.assertIn("optimization_technique", result)
        self.assertEqual(result["original_prompt"], "Create a web app")

    @patch('src.pipeline.prompt_optimizer.InferenceClient')
    def test_extract_proposal_returns_dict(self, mock_llm):
        """Test extract_proposal returns dict."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"title": "Test Project", "description": "A test project"}'))]
        )
        mock_llm.return_value = mock_instance

        optimizer = PromptOptimizer()
        result = optimizer.extract_proposal("Build a web application")

        self.assertIsInstance(result, dict)
        self.assertIn("title", result)

    @patch('src.pipeline.prompt_optimizer.InferenceClient')
    def test_enhance_for_diagram(self, mock_llm):
        """Test enhance_for_diagram returns enhanced prompt."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Enhanced prompt for workflow diagram"))]
        )
        mock_llm.return_value = mock_instance

        optimizer = PromptOptimizer()
        result = optimizer.enhance_for_diagram(
            diagram_type="workflow",
            plan="Build the app in 3 phases",
            user_prompt="Create a workflow",
        )

        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


class TestTimeAgent(unittest.TestCase):
    """Test cases for TimeAgent."""

    @patch('src.agents.time_agent.InferenceClient')
    def test_initialization(self, mock_llm):
        """Test TimeAgent initialization."""
        agent = TimeAgent()
        self.assertEqual(agent.name, "time_agent")
        mock_llm.assert_called_once()

    @patch('src.agents.time_agent.InferenceClient')
    def test_execute_success(self, mock_llm):
        """Test successful execution of TimeAgent."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""## 1. Project Phases & Milestones
- Phase 1: Design (2 weeks)
- Phase 2: Development (4 weeks)

## 2. Parallel Work Streams
- Frontend development
- Backend API

## 3. Gantt Chart
```mermaid
gantt
    title Test Project
    section Design
    UI Design :a1, 2026-05-10, 7d
    section Development
    Backend :after a1, 14d
```"""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = TimeAgent()
        state = {
            "proposal": "Create a website",
            "project_title": "Test Project",
            "team_size": 5,
            "timeline_weeks": 12,
        }

        result = agent.execute(state)

        self.assertIn("timetable", result)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["current_agent"], "time_agent")
        self.assertIn("milestones", result)
        self.assertIn("parallel_work_streams", result)

    @patch('src.agents.time_agent.InferenceClient')
    def test_execute_no_prompt(self, mock_llm):
        """Test TimeAgent execution with no prompt."""
        agent = TimeAgent()
        state = {
            "proposal": "",
            "prompt": "",
        }

        result = agent.execute(state)

        self.assertIsNotNone(result.get("error"))
        self.assertIn("No proposal", result["error"])

    @patch('src.agents.time_agent.InferenceClient')
    def test_execute_uses_fallback_prompt(self, mock_llm):
        """Test TimeAgent falls back to 'prompt' key if no 'proposal'."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Timetable content"))]
        )
        mock_llm.return_value = mock_instance

        agent = TimeAgent()
        state = {
            "prompt": "Create a website with auth",
        }

        result = agent.execute(state)
        self.assertIsNone(result.get("error"))


class TestPlanAgent(unittest.TestCase):
    """Test cases for PlanAgent."""

    @patch('src.agents.plan_agent.InferenceClient')
    def test_initialization(self, mock_llm):
        """Test PlanAgent initialization."""
        agent = PlanAgent()
        self.assertEqual(agent.name, "plan_agent")
        mock_llm.assert_called_once()

    @patch('src.agents.plan_agent.InferenceClient')
    def test_execute_success(self, mock_llm):
        """Test successful execution of PlanAgent."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Detailed plan with tasks and resources"))]
        )
        mock_llm.return_value = mock_instance

        agent = PlanAgent()
        state = {
            "timetable": "Phase 1: Design (1 week), Phase 2: Development (2 weeks)",
            "proposal": "Create a website",
            "team_size": 5,
            "priority": "medium",
        }

        result = agent.execute(state)

        self.assertIn("plan", result)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["current_agent"], "plan_agent")


class TestImageGeneratorAgent(unittest.TestCase):
    """Test cases for ImageGeneratorAgent - now supporting multiple diagram types."""

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_initialization(self, mock_llm):
        """Test ImageGeneratorAgent initialization."""
        agent = ImageGeneratorAgent()
        self.assertEqual(agent.name, "image_agent")
        mock_llm.assert_called_once()

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_execute_success_multi_diagram(self, mock_llm):
        """Test successful execution generating multiple diagrams."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""gantt
    title Test Gantt
    section Design
    Task1 :a1, 2026-05-10, 5d
    section Dev
    Task2 :after a1, 10d"""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ImageGeneratorAgent()
        state = {
            "plan": "Phase 1: Design, Phase 2: Development",
            "diagram_types": ["workflow", "ci_cd", "gantt"],
            "timetable": "Phase 1: Design (1 week)",
        }

        result = agent.execute(state)

        self.assertIn("diagrams", result)
        self.assertGreater(result["diagram_count"], 0)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["current_agent"], "image_agent")

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_generate_diagram_workflow(self, mock_llm):
        """Test generating a single workflow diagram."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Do something]
    B -->|No| D[Do another thing]"""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ImageGeneratorAgent()
        result = agent.generate_diagram(
            plan="Build a web app with login and dashboard",
            diagram_type=DiagramType.WORKFLOW,
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["diagram_type"], "workflow")
        self.assertIn("flowchart", result["mermaid_code"])

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_generate_diagram_system_design(self, mock_llm):
        """Test generating a system design diagram."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""flowchart TD
    Client --> API[API Gateway]
    API --> Auth[Auth Service]
    API --> Core[Core Service]
    Core --> DB[(Database)]"""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ImageGeneratorAgent()
        result = agent.generate_diagram(
            plan="Build a microservices architecture with API gateway, auth, and core services",
            diagram_type=DiagramType.SYSTEM_DESIGN,
        )

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["diagram_type"], "system_design")

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_execute_no_plan(self, mock_llm):
        """Test ImageGeneratorAgent execution with no plan."""
        agent = ImageGeneratorAgent()
        state = {"plan": "", "diagram_types": ["workflow"]}

        result = agent.execute(state)

        self.assertEqual(result.get("diagram_count", 0), 0)
        self.assertIn("error", result)

    def test_validate_diagram_invalid(self):
        """Test that invalid diagrams are properly flagged."""
        agent = ImageGeneratorAgent()
        result = agent._validate_diagram("not a diagram")
        self.assertFalse(result)

    def test_validate_diagram_valid(self):
        """Test that valid Mermaid diagrams pass validation."""
        agent = ImageGeneratorAgent()
        result = agent._validate_diagram("gantt\ntitle Test\nsection A\nTask :a1, 2026-05-10, 5d")
        self.assertTrue(result)

    def test_diagram_type_enum(self):
        """Test DiagramType enum values."""
        self.assertEqual(DiagramType.WORKFLOW.value, "workflow")
        self.assertEqual(DiagramType.CI_CD.value, "ci_cd")
        self.assertEqual(DiagramType.SYSTEM_DESIGN.value, "system_design")
        self.assertEqual(DiagramType.FLOWCHART.value, "flowchart")
        self.assertEqual(DiagramType.ARCHITECTURE.value, "architecture")
        self.assertEqual(DiagramType.GANTT.value, "gantt")

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_all_diagram_templates_exist(self, mock_llm):
        """Test that templates exist for all diagram types."""
        agent = ImageGeneratorAgent()
        for diagram_type in DiagramType:
            self.assertIn(diagram_type, agent.DIAGRAM_TEMPLATES)
            template = agent.DIAGRAM_TEMPLATES[diagram_type]
            self.assertIn("system_role", template)
            self.assertIn("requirements", template)


class TestBaseValidatorAgent(unittest.TestCase):
    """Test cases for BaseValidatorAgent abstract base class."""

    def test_base_class_is_abstract(self):
        """Test that BaseValidatorAgent cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseValidatorAgent("test_validator")

    def test_inheritance_structure(self):
        """Test that ValidatorAgent inherits from BaseValidatorAgent."""
        self.assertTrue(issubclass(ValidatorAgent, BaseValidatorAgent))
        self.assertTrue(issubclass(TimeValidatorAgent, BaseValidatorAgent))

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_basic_mermaid_validation_passes(self, mock_llm):
        """Test basic Mermaid validation passes for valid diagrams."""

        class TestValidator(BaseValidatorAgent):
            def _get_model(self): return "test/model"
            def execute(self, state): return state

        agent = TestValidator("test_validator")
        result = agent._basic_mermaid_validation(
            "gantt\ntitle Test\nsection A\nTask :a1, 2026-05-10, 5d"
        )
        self.assertTrue(result["is_valid"])

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_basic_mermaid_validation_fails_empty(self, mock_llm):
        """Test basic Mermaid validation fails for empty input."""

        class TestValidator(BaseValidatorAgent):
            def _get_model(self): return "test/model"
            def execute(self, state): return state

        agent = TestValidator("test_validator")
        result = agent._basic_mermaid_validation("")
        self.assertFalse(result["is_valid"])
        self.assertIn("Empty", result["feedback"])

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_basic_mermaid_validation_fails_no_mermaid(self, mock_llm):
        """Test basic Mermaid validation fails for non-Mermaid text."""

        class TestValidator(BaseValidatorAgent):
            def _get_model(self): return "test/model"
            def execute(self, state): return state

        agent = TestValidator("test_validator")
        result = agent._basic_mermaid_validation("This is just text")
        self.assertFalse(result["is_valid"])
        self.assertIn("valid Mermaid", result["feedback"])

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_basic_mermaid_validation_fails_brackets(self, mock_llm):
        """Test basic Mermaid validation fails for mismatched brackets."""

        class TestValidator(BaseValidatorAgent):
            def _get_model(self): return "test/model"
            def execute(self, state): return state

        agent = TestValidator("test_validator")
        result = agent._basic_mermaid_validation("gantt\ntitle [Unclosed")
        self.assertFalse(result["is_valid"])
        self.assertIn("Mismatched", result["feedback"])

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_parse_validation_response(self, mock_llm):
        """Test parsing of LLM validation response."""

        class TestValidator(BaseValidatorAgent):
            def _get_model(self): return "test/model"
            def execute(self, state): return state

        agent = TestValidator("test_validator")
        response = """VALID: true
CRITICAL_ISSUES: None
WARNINGS: Minor suggestion about naming
SUGGESTIONS: Use more descriptive node names
FEEDBACK: Overall the diagram is well-formed and logically consistent."""
        result = agent._parse_llm_validation(response)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["feedback"], "Overall the diagram is well-formed and logically consistent.")

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_parse_validation_response_invalid(self, mock_llm):
        """Test parsing of invalid LLM validation response."""

        class TestValidator(BaseValidatorAgent):
            def _get_model(self): return "test/model"
            def execute(self, state): return state

        agent = TestValidator("test_validator")
        response = """VALID: false
CRITICAL_ISSUES: Circular dependency detected
WARNINGS: Node naming could be improved
SUGGESTIONS: None
FEEDBACK: The diagram has a circular reference."""
        result = agent._parse_llm_validation(response)
        self.assertFalse(result["is_valid"])
        self.assertIn("Circular dependency", result["critical_issues"][0])


class TestValidatorAgent(BaseValidatorAgent):
    """Concrete subclass for testing ValidatorAgent's re-routing logic without LLM."""

    def test_routing_on_validation_failure(self):
        """Test that failed validation sets route_to to image_agent."""
        agent = ValidatorAgent()
        # We bypass the LLM by directly checking the state logic
        state = {
            "diagrams": [
                {
                    "mermaid_code": "invalid diagram with no proper syntax",
                    "diagram_type": "workflow",
                }
            ],
            "previous_prompts": ["Generate a workflow diagram for a web app"],
        }
        # Run execute - even if LLM fails, basic validation should catch it
        result = agent.execute(state)

        # Should not be valid since basic mermaid validation fails
        self.assertFalse(result["overall_validation"])
        self.assertIn("route_to", result)

    @patch('src.agents.validator_agent.InferenceClient')
    def test_routing_on_validation_success(self, mock_llm):
        """Test that successful validation sets route_to to end."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""VALID: true
CRITICAL_ISSUES: None
WARNINGS: None
SUGGESTIONS: None
FEEDBACK: Diagram is well-formed."""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ValidatorAgent()
        state = {
            "diagrams": [
                {
                    "mermaid_code": "gantt\ntitle Test\nsection A\nTask :a1, 2026-05-10, 5d",
                    "diagram_type": "workflow",
                }
            ],
            "previous_prompts": [],
        }

        result = agent.execute(state)

        self.assertTrue(result["overall_validation"])
        self.assertEqual(result["route_to"], "end")

    @patch('src.agents.validator_agent.InferenceClient')
    def test_improvement_prompt_on_failure(self, mock_llm):
        """Test that improvement prompt is generated when validation fails."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""VALID: false
CRITICAL_ISSUES: Missing diagram type keyword
WARNINGS: Node labels are unclear
SUGGESTIONS: Use descriptive labels
FEEDBACK: Diagram lacks proper Mermaid syntax."""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ValidatorAgent()
        state = {
            "diagrams": [
                {
                    "mermaid_code": "some bad mermaid code",
                    "diagram_type": "workflow",
                }
            ],
            "previous_prompts": [
                "Generate a workflow diagram for e-commerce"
            ],
        }

        result = agent.execute(state)

        self.assertFalse(result["overall_validation"])
        self.assertIn("improvement_prompt", result)
        self.assertIn("improvement", result["improvement_prompt"].lower())
        self.assertIn("fixes", result["improvement_prompt"].lower()
                      or "IMPORTANT" in result["improvement_prompt"])


class TestTimeValidatorAgent(unittest.TestCase):
    """Test cases for TimeValidatorAgent."""

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_initialization(self, mock_llm):
        """Test TimeValidatorAgent initialization."""
        agent = TimeValidatorAgent()
        self.assertEqual(agent.name, "time_validator_agent")
        mock_llm.assert_called_once()

    def test_inherits_from_base(self):
        """Test TimeValidatorAgent inherits from BaseValidatorAgent."""
        self.assertTrue(issubclass(TimeValidatorAgent, BaseValidatorAgent))

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_validate_timetable_passes(self, mock_llm):
        """Test timetable validation passes for valid timetable."""
        agent = TimeValidatorAgent()
        result = agent._validate_timetable(
            "gantt\ntitle Test\nsection Design\nUI :a1, 2026-05-10, 7d\nsection Dev\nBackend :after a1, 14d"
        )
        self.assertTrue(result["is_valid"])

    def test_validate_timetable_empty(self):
        """Test timetable validation fails for empty input."""
        agent = TimeValidatorAgent()
        result = agent._validate_timetable("")
        self.assertFalse(result["is_valid"])
        self.assertIn("empty", result["feedback"].lower())

    def test_validate_timetable_insufficient(self):
        """Test timetable validation fails for insufficient content."""
        agent = TimeValidatorAgent()
        result = agent._validate_timetable("short")
        self.assertFalse(result["is_valid"])

    def test_validate_timetable_no_phases(self):
        """Test timetable validation fails when no phases detected."""
        agent = TimeValidatorAgent()
        result = agent._validate_timetable(
            "some random text without any project phases"
        )
        self.assertFalse(result["is_valid"])
        self.assertIn("phases", result["feedback"].lower())

    def test_validate_milestones_passes(self):
        """Test milestones validation passes for valid milestones."""
        agent = TimeValidatorAgent()
        result = agent._validate_milestones([
            "Design complete",
            "Development complete",
            "Testing complete",
            "Deployment complete"
        ])
        self.assertTrue(result["is_valid"])

    def test_validate_milestones_empty(self):
        """Test milestones validation fails for empty list."""
        agent = TimeValidatorAgent()
        result = agent._validate_milestones([])
        self.assertFalse(result["is_valid"])

    def test_validate_milestones_insufficient(self):
        """Test milestones validation fails for too few milestones."""
        agent = TimeValidatorAgent()
        result = agent._validate_milestones(["Design"])
        self.assertFalse(result["is_valid"])

    def test_validate_parallel_streams_passes(self):
        """Test parallel streams validation passes for valid streams."""
        agent = TimeValidatorAgent()
        result = agent._validate_parallel_streams([
            "Frontend development",
            "Backend API development"
        ])
        self.assertTrue(result["is_valid"])

    def test_validate_parallel_streams_empty(self):
        """Test parallel streams validation fails for empty list."""
        agent = TimeValidatorAgent()
        result = agent._validate_parallel_streams([])
        self.assertFalse(result["is_valid"])

    def test_validate_parallel_streams_insufficient(self):
        """Test parallel streams validation fails for vague content."""
        agent = TimeValidatorAgent()
        result = agent._validate_parallel_streams(["a", "b"])
        self.assertFalse(result["is_valid"])

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_execute_success(self, mock_llm):
        """Test successful execution of TimeValidatorAgent."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""VALID: true
CRITICAL_ISSUES: None
WARNINGS: None
SUGGESTIONS: None
FEEDBACK: Timeline is well-structured."""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = TimeValidatorAgent()
        state = {
            "timetable": "gantt\ntitle Test\nsection Design\nUI :a1, 2026-05-10, 7d\nsection Dev\nBackend :after a1, 14d",
            "milestones": ["Design complete", "Dev complete", "Testing complete"],
            "parallel_work_streams": ["Frontend", "Backend"],
        }

        result = agent.execute(state)

        self.assertTrue(result["time_overall_validation"])
        self.assertEqual(result["route_to"], "plan_agent")
        self.assertEqual(result["current_agent"], "time_validator_agent")

    @patch('src.agents.base_validator_agent.InferenceClient')
    def test_execute_failure_routes_to_time_agent(self, mock_llm):
        """Test that failed time validation routes back to time_agent."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""VALID: false
CRITICAL_ISSUES: Missing deployment phase
WARNINGS: No parallel streams identified
SUGGESTIONS: Add deployment phase, identify parallel work
FEEDBACK: Timeline lacks proper structure."""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = TimeValidatorAgent()
        state = {
            "timetable": "gantt\ntitle Test\nsection Design\nUI :a1, 2026-05-10, 7d",
            "milestones": ["Design"],
            "parallel_work_streams": [],
        }

        result = agent.execute(state)

        self.assertFalse(result["time_overall_validation"])
        self.assertIn("corrective_prompt", result)
        # route_to should be time_agent only if retries not exhausted
        # Without retry count, defaults to 0 < MAX, so route to time_agent
        self.assertIn(result.get("route_to"), ["time_agent", "plan_agent"])

    def test_corrective_prompt_generation(self):
        """Test that corrective prompts are properly generated."""
        agent = TimeValidatorAgent()
        validation_results = [
            {
                "field": "timetable",
                "is_valid": False,
                "feedback": "Too few phases",
                "critical_issues": ["Missing deployment phase"],
                "warnings": [],
                "suggestions": ["Add a deployment phase with tasks"],
            },
            {
                "field": "milestones",
                "is_valid": True,
                "feedback": "Adequate milestones",
                "critical_issues": [],
                "warnings": [],
                "suggestions": [],
            },
        ]

        corrective = agent._build_corrective_prompt(
            validation_results,
            "gantt\ntitle Test\nsection Design\nUI :a1, 2026-05-10, 7d"
        )

        self.assertIn("IMPORTANT", corrective)
        self.assertIn("deployment", corrective.lower())
        self.assertIn("TIMETABLE", corrective)


class TestAudienceType(unittest.TestCase):
    """Test cases for the AudienceType enum."""

    def test_enum_values(self):
        """Test that AudienceType has expected values."""
        self.assertEqual(AudienceType.ENGINEER.value, "engineer")
        self.assertEqual(AudienceType.STAKEHOLDER.value, "stakeholder")

    def test_enum_from_string_engineer(self):
        """Test constructing AudienceType from string."""
        self.assertEqual(AudienceType("engineer"), AudienceType.ENGINEER)

    def test_enum_from_string_stakeholder(self):
        """Test constructing AudienceType from string."""
        self.assertEqual(AudienceType("stakeholder"), AudienceType.STAKEHOLDER)

    def test_enum_invalid_value(self):
        """Test that invalid audience type raises ValueError."""
        with self.assertRaises(ValueError):
            AudienceType("manager")

    def test_enum_is_string(self):
        """Test AudienceType is a str enum — value serialises as plain string."""
        self.assertIsInstance(AudienceType.ENGINEER, str)
        # .value gives the raw string; str() on a str-enum member returns the value in 3.11
        self.assertEqual(AudienceType.STAKEHOLDER.value, "stakeholder")


class TestSchemaModels(unittest.TestCase):
    """Test cases for Pydantic schema models."""

    def test_prompt_request_valid(self):
        """Test PromptRequest with valid data."""
        prompt = PromptRequest(
            user_prompt="Generate a comprehensive workflow diagram",
            diagram_types=[DiagramType.WORKFLOW, DiagramType.CI_CD],
            optimize_prompt=True,
            include_gantt=True,
            priority="high",
        )
        self.assertEqual(len(prompt.diagram_types), 2)
        self.assertTrue(prompt.optimize_prompt)
        self.assertEqual(prompt.priority, "high")

    def test_prompt_request_default_diagram_types(self):
        """Test PromptRequest default diagram types."""
        prompt = PromptRequest(
            user_prompt="Generate diagrams for my project",
        )
        self.assertEqual(len(prompt.diagram_types), 2)
        self.assertEqual(prompt.diagram_types[0], DiagramType.WORKFLOW)
        self.assertEqual(prompt.diagram_types[1], DiagramType.CI_CD)

    def test_prompt_request_default_audience_is_engineer(self):
        """Test that audience_type defaults to engineer."""
        prompt = PromptRequest(user_prompt="Generate diagrams for my project")
        self.assertEqual(prompt.audience_type, AudienceType.ENGINEER)

    def test_prompt_request_stakeholder_audience(self):
        """Test explicitly setting audience_type to stakeholder."""
        prompt = PromptRequest(
            user_prompt="Generate stakeholder presentation",
            audience_type="stakeholder",
        )
        self.assertEqual(prompt.audience_type, AudienceType.STAKEHOLDER)

    def test_prompt_request_engineer_audience(self):
        """Test explicitly setting audience_type to engineer."""
        prompt = PromptRequest(
            user_prompt="Generate technical diagrams",
            audience_type="engineer",
        )
        self.assertEqual(prompt.audience_type, AudienceType.ENGINEER)

    def test_prompt_request_invalid_audience_raises(self):
        """Test that an unrecognised audience_type is rejected."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PromptRequest(
                user_prompt="Generate diagrams",
                audience_type="cto",
            )


class TestWorkflow(unittest.TestCase):
    """Test cases for the workflow."""

    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.TimeValidatorAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_create_workflow(self, mock_validator, mock_image, mock_plan,
                              mock_time_validator, mock_time):
        """Test workflow creation."""
        mock_time.return_value.execute = Mock(return_value={"timetable": "test"})
        mock_time_validator.return_value.execute = Mock(
            return_value={"time_overall_validation": True}
        )
        mock_plan.return_value.execute = Mock(return_value={"plan": "test"})
        mock_image.return_value.execute = Mock(return_value={"diagrams": []})
        mock_validator.return_value.execute = Mock(
            return_value={"validation_results": []}
        )

        workflow = create_flowforge_workflow()

        self.assertIsNotNone(workflow)
        mock_time.assert_called_once()
        mock_time_validator.assert_called_once()
        mock_plan.assert_called_once()
        mock_image.assert_called_once()
        mock_validator.assert_called_once()

    @patch('src.workflow.graph_workflow.SessionManager')
    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.TimeValidatorAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_workflow_state_flow(
        self,
        mock_validator,
        mock_image,
        mock_plan,
        mock_time_validator,
        mock_time,
        mock_session_manager,
    ):
        """Test that workflow state flows correctly through agents."""

        # Time Agent mock
        time_instance = Mock()
        time_instance.execute.return_value = {
            "timetable": "Phase 1: Design, Phase 2: Dev",
            "milestones": ["Design complete", "Development complete"],
            "parallel_work_streams": ["Frontend", "Backend"],
            "error": None,
        }
        mock_time.return_value = time_instance

        # Time Validator mock
        time_validator_instance = Mock()
        time_validator_instance.execute.return_value = {
            "time_overall_validation": True,
            "time_validation_results": [],
            "error": None,
        }
        mock_time_validator.return_value = time_validator_instance

        # Plan Agent mock
        plan_instance = Mock()
        plan_instance.execute.return_value = {
            "plan": "Task breakdown with resources",
            "error": None,
        }
        mock_plan.return_value = plan_instance

        # Image Agent mock
        image_instance = Mock()
        image_instance.execute.return_value = {
            "diagrams": [
                {
                    "diagram_type": "workflow",
                    "mermaid_code": "flowchart TD\n    A --> B",
                    "is_valid": True,
                }
            ],
            "diagram_count": 1,
            "valid_diagram_count": 1,
            "error": None,
        }
        mock_image.return_value = image_instance

        # Validator Agent mock
        validator_instance = Mock()
        validator_instance.execute.return_value = {
            "validation_results": [{"is_valid": True}],
            "overall_validation": True,
            "valid_count": 1,
            "total_count": 1,
            "error": None,
        }
        mock_validator.return_value = validator_instance

        from src.workflow.graph_workflow import run_flowforge_workflow

        result = run_flowforge_workflow(
            proposal="Build a web app",
            prompt="Generate workflow diagram",
            hf_token="test-token",
            optimize_prompt=False,
        )

        self.assertIsNotNone(result)
        self.assertIn("diagrams", result)
        self.assertIn("overall_validation", result)

    @patch('src.workflow.graph_workflow.SessionManager')
    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.TimeValidatorAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_workflow_with_validation_retry(
        self,
        mock_validator,
        mock_image,
        mock_plan,
        mock_time_validator,
        mock_time,
        mock_session_manager,
    ):
        """Test workflow where validation fails first time then succeeds on retry."""

        # Time Agent
        time_instance = Mock()
        time_instance.execute.return_value = {
            "timetable": "Phase 1: Design",
            "milestones": ["Design"],
            "parallel_work_streams": [],
            "error": None,
        }
        mock_time.return_value = time_instance

        # Time Validator - fails first, passes second
        time_validator_instance = Mock()
        time_validator_instance.execute.side_effect = [
            {
                "time_overall_validation": False,
                "time_validation_results": [
                    {"field": "milestones", "is_valid": False, "feedback": "Too few"}
                ],
                "corrective_prompt": "Add more milestones",
                "route_to": "time_agent",
                "error": "Time validation failed",
            },
            {
                "time_overall_validation": True,
                "time_validation_results": [],
                "route_to": "plan_agent",
                "error": None,
            },
        ]
        mock_time_validator.return_value = time_validator_instance

        # Time Agent - second call (retry)
        time_instance_retry = Mock()
        time_instance_retry.execute.return_value = {
            "timetable": "Phase 1: Design, Phase 2: Dev",
            "milestones": ["Design complete", "Development complete"],
            "parallel_work_streams": ["Frontend", "Backend"],
            "error": None,
        }
        mock_time.return_value = time_instance_retry

        # Plan Agent
        plan_instance = Mock()
        plan_instance.execute.return_value = {
            "plan": "Detailed plan",
            "error": None,
        }
        mock_plan.return_value = plan_instance

        # Image Agent
        image_instance = Mock()
        image_instance.execute.return_value = {
            "diagrams": [
                {
                    "diagram_type": "workflow",
                    "mermaid_code": "flowchart TD\n    A --> B",
                    "is_valid": True,
                }
            ],
            "diagram_count": 1,
            "valid_diagram_count": 1,
            "error": None,
        }
        mock_image.return_value = image_instance

        # Validator Agent
        validator_instance = Mock()
        validator_instance.execute.return_value = {
            "validation_results": [{"is_valid": True}],
            "overall_validation": True,
            "valid_count": 1,
            "total_count": 1,
            "error": None,
        }
        mock_validator.return_value = validator_instance

        from src.workflow.graph_workflow import run_flowforge_workflow

        result = run_flowforge_workflow(
            proposal="Build a web app",
            prompt="Generate workflow diagram",
            hf_token="test-token",
            optimize_prompt=False,
        )

        self.assertIsNotNone(result)
        self.assertIn("diagrams", result)

    @patch('src.workflow.graph_workflow.SessionManager')
    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.TimeValidatorAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_workflow_with_diagram_validation_retry(
        self,
        mock_validator,
        mock_image,
        mock_plan,
        mock_time_validator,
        mock_time,
        mock_session_manager,
    ):
        """Test workflow where diagram validation fails then succeeds on retry."""

        # Time Agent
        time_instance = Mock()
        time_instance.execute.return_value = {
            "timetable": "gantt\ntitle Test\nsection Design\nUI :a1, 2026-05-10, 7d",
            "milestones": ["Design complete"],
            "parallel_work_streams": ["Frontend"],
            "error": None,
        }
        mock_time.return_value = time_instance

        # Time Validator passes
        time_validator_instance = Mock()
        time_validator_instance.execute.return_value = {
            "time_overall_validation": True,
            "route_to": "plan_agent",
            "error": None,
        }
        mock_time_validator.return_value = time_validator_instance

        # Plan Agent
        plan_instance = Mock()
        plan_instance.execute.return_value = {
            "plan": "Detailed plan",
            "error": None,
        }
        mock_plan.return_value = plan_instance

        # Image Agent
        image_instance = Mock()
        image_instance.execute.return_value = {
            "diagrams": [
                {
                    "diagram_type": "workflow",
                    "mermaid_code": "flowchart TD\n    A --> B",
                    "is_valid": True,
                }
            ],
            "diagram_count": 1,
            "valid_diagram_count": 1,
            "error": None,
        }
        mock_image.return_value = image_instance

        # Validator fails first, passes on retry
        validator_instance = Mock()
        validator_instance.execute.side_effect = [
            {
                "validation_results": [
                    {"is_valid": False, "diagram_type": "workflow"}
                ],
                "overall_validation": False,
                "valid_count": 0,
                "total_count": 1,
                "error": "1/1 diagrams failed validation",
                "improvement_prompt": "Fix the workflow diagram",
                "route_to": "image_agent",
            },
            {
                "validation_results": [
                    {"is_valid": True, "diagram_type": "workflow"}
                ],
                "overall_validation": True,
                "valid_count": 1,
                "total_count": 1,
                "error": None,
                "route_to": "end",
            },
        ]
        mock_validator.return_value = validator_instance

        from src.workflow.graph_workflow import run_flowforge_workflow

        result = run_flowforge_workflow(
            proposal="Build a web app",
            prompt="Generate workflow diagram",
            hf_token="test-token",
            optimize_prompt=False,
        )

        self.assertIsNotNone(result)
        self.assertIn("overall_validation", result)

    def test_flowforge_state_as_dict(self):
        """Test FlowForgeState behaves as a dict."""
        from src.workflow.graph_workflow import FlowForgeState

        state = FlowForgeState(
            proposal="Test proposal",
            prompt="Test prompt",
            hf_token="test",
        )

        # Dict-style access
        self.assertEqual(state["proposal"], "Test proposal")

        # Attribute-style access
        self.assertEqual(state.proposal, "Test proposal")

        # Setting values
        state.timetable = "Phase 1"
        self.assertEqual(state["timetable"], "Phase 1")


class TestStakeholderDiagramAgent(unittest.TestCase):
    """Tests for stakeholder-specific diagram generation behaviour."""

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_stakeholder_templates_exist_for_all_required_types(self, mock_llm):
        """Stakeholder templates must exist for flowchart, gantt, architecture."""
        agent = ImageGeneratorAgent()
        for dt in [DiagramType.FLOWCHART, DiagramType.GANTT, DiagramType.ARCHITECTURE]:
            self.assertIn(dt, agent.STAKEHOLDER_DIAGRAM_TEMPLATES)
            tmpl = agent.STAKEHOLDER_DIAGRAM_TEMPLATES[dt]
            self.assertIn("system_role", tmpl)
            self.assertIn("requirements", tmpl)

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_stakeholder_template_system_role_is_business_language(self, mock_llm):
        """Stakeholder flowchart role should mention stakeholders / business."""
        agent = ImageGeneratorAgent()
        role = agent.STAKEHOLDER_DIAGRAM_TEMPLATES[DiagramType.FLOWCHART]["system_role"].lower()
        self.assertTrue(
            "stakeholder" in role or "business" in role,
            "Stakeholder system_role should reference business audience",
        )

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_stakeholder_diagram_types_set_contains_correct_members(self, mock_llm):
        """STAKEHOLDER_DIAGRAM_TYPES should include only business-friendly types."""
        agent = ImageGeneratorAgent()
        expected = {DiagramType.FLOWCHART, DiagramType.GANTT, DiagramType.ARCHITECTURE}
        self.assertEqual(agent.STAKEHOLDER_DIAGRAM_TYPES, expected)

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_build_prompt_uses_stakeholder_template_when_audience_is_stakeholder(self, mock_llm):
        """_build_diagram_prompt should return stakeholder template content for stakeholder audience."""
        agent = ImageGeneratorAgent()
        prompt = agent._build_diagram_prompt(
            plan="Build a customer portal",
            diagram_type=DiagramType.FLOWCHART,
            timeline=None,
            audience_type="stakeholder",
        )
        stakeholder_role = agent.STAKEHOLDER_DIAGRAM_TEMPLATES[DiagramType.FLOWCHART]["system_role"]
        self.assertIn(stakeholder_role[:30], prompt)

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_build_prompt_uses_engineer_template_when_audience_is_engineer(self, mock_llm):
        """_build_diagram_prompt should return engineer template content for engineer audience."""
        agent = ImageGeneratorAgent()
        prompt = agent._build_diagram_prompt(
            plan="Build a customer portal",
            diagram_type=DiagramType.FLOWCHART,
            timeline=None,
            audience_type="engineer",
        )
        engineer_role = agent.DIAGRAM_TEMPLATES[DiagramType.FLOWCHART]["system_role"]
        self.assertIn(engineer_role[:30], prompt)

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_execute_filters_engineer_diagrams_for_stakeholder(self, mock_llm):
        """execute() must keep only stakeholder-valid diagram types when audience=stakeholder."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="graph TD\n    A[Start] --> B[End]"
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ImageGeneratorAgent()
        state = {
            "plan": "Build a portal",
            # Engineer diagram types requested — should be filtered down
            "diagram_types": ["workflow", "ci_cd", "system_design", "flowchart", "gantt"],
            "timetable": "Phase 1",
            "audience_type": "stakeholder",
        }
        result = agent.execute(state)

        for diagram in result["diagrams"]:
            self.assertIn(
                diagram["diagram_type"],
                ["flowchart", "gantt", "architecture"],
                "Stakeholder output must not contain technical engineer-only diagram types",
            )

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_execute_uses_all_requested_diagrams_for_engineer(self, mock_llm):
        """execute() must not filter diagram types when audience=engineer."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="graph TD\n    A[Start] --> B[End]"
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ImageGeneratorAgent()
        requested = ["workflow", "ci_cd"]
        state = {
            "plan": "Build a portal",
            "diagram_types": requested,
            "timetable": "Phase 1",
            "audience_type": "engineer",
        }
        result = agent.execute(state)
        returned_types = [d["diagram_type"] for d in result["diagrams"]]
        for dt in requested:
            self.assertIn(dt, returned_types)

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_execute_falls_back_to_flowchart_gantt_when_no_stakeholder_types_requested(self, mock_llm):
        """If stakeholder mode is active but no stakeholder diagram types were selected, use defaults."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="graph TD\n    A[Start] --> B[End]"
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ImageGeneratorAgent()
        state = {
            "plan": "Build a portal",
            "diagram_types": ["workflow", "ci_cd", "system_design"],  # none in stakeholder set
            "timetable": "Phase 1",
            "audience_type": "stakeholder",
        }
        result = agent.execute(state)
        # Should have generated diagrams using fallback types
        self.assertGreater(result["diagram_count"], 0)
        for diagram in result["diagrams"]:
            self.assertIn(diagram["diagram_type"], ["flowchart", "gantt", "architecture"])

    @patch('src.agents.image_generator_agent.InferenceClient')
    def test_generate_diagram_passes_audience_type_to_prompt(self, mock_llm):
        """generate_diagram() with audience_type=stakeholder should use stakeholder prompt."""
        mock_instance = Mock()
        captured_messages = []

        def capture_call(messages, max_tokens):
            captured_messages.extend(messages)
            return MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content="graph TD\n    A[Phase] --> B[Done]"
                ))]
            )

        mock_instance.chat_completion.side_effect = capture_call
        mock_llm.return_value = mock_instance

        agent = ImageGeneratorAgent()
        agent.generate_diagram(
            plan="Build customer portal",
            diagram_type=DiagramType.FLOWCHART,
            audience_type="stakeholder",
        )

        self.assertTrue(len(captured_messages) > 0)
        prompt_content = captured_messages[0]["content"]
        stakeholder_role = agent.STAKEHOLDER_DIAGRAM_TEMPLATES[DiagramType.FLOWCHART]["system_role"]
        self.assertIn(stakeholder_role[:20], prompt_content)


class TestStakeholderPDF(unittest.TestCase):
    """Tests for the stakeholder PDF builder."""

    def _make_diagram(self, diagram_type: str, include_image: bool = True) -> dict:
        """Build a minimal diagram dict with a tiny real PNG."""
        import base64
        from PIL import Image as PILImage
        from io import BytesIO

        image_data = None
        if include_image:
            img = PILImage.new("RGB", (100, 50), color=(200, 200, 200))
            buf = BytesIO()
            img.save(buf, format="PNG")
            image_data = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return {
            "diagram_type": diagram_type,
            "title": diagram_type.replace("_", " ").title(),
            "image_data": image_data,
            "is_valid": True,
        }

    def _make_proposal(self, **overrides) -> dict:
        base = {
            "title": "Customer Portal",
            "description": "A self-service customer portal for account management.",
            "requirements": ["User login", "Dashboard", "Reports"],
            "constraints": ["Go live within 6 months"],
            "timeline_weeks": 24,
            "team_size": 6,
            "budget_range": "$50k-$80k",
            "tech_stack": ["React", "FastAPI"],
            "priority": "high",
        }
        base.update(overrides)
        return base

    def test_returns_bytes_io(self):
        """build_stakeholder_pdf should return a BytesIO buffer."""
        import sys; sys.path.insert(0, 'frontend')
        from document_pdf import build_stakeholder_pdf
        from io import BytesIO

        diagrams = [self._make_diagram("flowchart")]
        buf = build_stakeholder_pdf(diagrams, self._make_proposal())
        self.assertIsInstance(buf, BytesIO)

    def test_buffer_is_non_empty(self):
        """PDF buffer should contain data."""
        import sys; sys.path.insert(0, 'frontend')
        from document_pdf import build_stakeholder_pdf

        diagrams = [self._make_diagram("flowchart"), self._make_diagram("gantt")]
        buf = build_stakeholder_pdf(diagrams, self._make_proposal())
        content = buf.read()
        self.assertGreater(len(content), 0)

    def test_buffer_starts_with_pdf_magic_bytes(self):
        """Generated file should be a valid PDF (starts with %PDF)."""
        import sys; sys.path.insert(0, 'frontend')
        from document_pdf import build_stakeholder_pdf

        diagrams = [self._make_diagram("flowchart")]
        buf = build_stakeholder_pdf(diagrams, self._make_proposal())
        self.assertTrue(buf.read(4) == b"%PDF")

    def test_diagram_without_image_is_skipped(self):
        """Diagrams with no image_data should not crash the builder."""
        import sys; sys.path.insert(0, 'frontend')
        from document_pdf import build_stakeholder_pdf

        diagrams = [
            self._make_diagram("flowchart", include_image=False),
            self._make_diagram("gantt", include_image=True),
        ]
        buf = build_stakeholder_pdf(diagrams, self._make_proposal())
        self.assertGreater(len(buf.read()), 0)

    def test_empty_diagram_list_still_builds(self):
        """An empty diagram list should produce a spec-only PDF without crashing."""
        import sys; sys.path.insert(0, 'frontend')
        from document_pdf import build_stakeholder_pdf

        buf = build_stakeholder_pdf([], self._make_proposal())
        self.assertGreater(len(buf.read()), 0)

    def test_missing_optional_proposal_fields(self):
        """Missing optional fields (requirements, constraints, etc.) should not crash."""
        import sys; sys.path.insert(0, 'frontend')
        from document_pdf import build_stakeholder_pdf

        proposal = {"title": "Minimal Project", "description": "Minimal description here."}
        diagrams = [self._make_diagram("flowchart")]
        buf = build_stakeholder_pdf(diagrams, proposal)
        self.assertGreater(len(buf.read()), 0)

    def test_all_three_stakeholder_diagram_types_rendered(self):
        """All three stakeholder diagram types should be included without error."""
        import sys; sys.path.insert(0, 'frontend')
        from document_pdf import build_stakeholder_pdf

        diagrams = [
            self._make_diagram("flowchart"),
            self._make_diagram("gantt"),
            self._make_diagram("architecture"),
        ]
        buf = build_stakeholder_pdf(diagrams, self._make_proposal())
        self.assertTrue(buf.read(4) == b"%PDF")


class TestEmailSender(unittest.TestCase):
    """Tests for the email sender module."""

    def setUp(self):
        import sys
        sys.path.insert(0, 'frontend')

    def test_raises_value_error_when_smtp_not_configured(self):
        """send_pdf_email should raise ValueError if SMTP config is missing."""
        from email_sender import send_pdf_email
        from unittest.mock import patch
        from io import BytesIO

        with patch('email_sender.Config') as mock_cfg:
            mock_cfg.SMTP_HOST = ""
            mock_cfg.SMTP_USER = ""
            mock_cfg.SMTP_PASSWORD = ""
            mock_cfg.SMTP_PORT = 587
            mock_cfg.SMTP_USE_TLS = True

            with self.assertRaises(ValueError) as ctx:
                send_pdf_email(
                    recipient="user@example.com",
                    pdf_buffer=BytesIO(b"%PDF-fake"),
                    project_title="Test",
                    audience_type="engineer",
                )
            self.assertIn("SMTP", str(ctx.exception))

    @patch('email_sender.smtplib.SMTP')
    def test_sends_email_when_smtp_configured(self, mock_smtp_cls):
        """send_pdf_email should call SMTP login and sendmail when configured."""
        from email_sender import send_pdf_email
        from unittest.mock import patch, MagicMock
        from io import BytesIO

        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = Mock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = Mock(return_value=False)

        with patch('email_sender.Config') as mock_cfg:
            mock_cfg.SMTP_HOST = "smtp.example.com"
            mock_cfg.SMTP_USER = "sender@example.com"
            mock_cfg.SMTP_PASSWORD = "secret"
            mock_cfg.SMTP_PORT = 587
            mock_cfg.SMTP_USE_TLS = False

            send_pdf_email(
                recipient="user@example.com",
                pdf_buffer=BytesIO(b"%PDF-fake"),
                project_title="Test Project",
                audience_type="stakeholder",
            )

        mock_server.login.assert_called_once_with("sender@example.com", "secret")
        mock_server.sendmail.assert_called_once()

    @patch('email_sender.smtplib.SMTP')
    def test_subject_reflects_audience_type_stakeholder(self, mock_smtp_cls):
        """Email subject should say 'Stakeholder Report' for stakeholder audience."""
        from email_sender import send_pdf_email
        from unittest.mock import patch, MagicMock
        from io import BytesIO

        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = Mock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = Mock(return_value=False)

        with patch('email_sender.Config') as mock_cfg:
            mock_cfg.SMTP_HOST = "smtp.example.com"
            mock_cfg.SMTP_USER = "sender@example.com"
            mock_cfg.SMTP_PASSWORD = "secret"
            mock_cfg.SMTP_PORT = 587
            mock_cfg.SMTP_USE_TLS = False

            send_pdf_email(
                recipient="user@example.com",
                pdf_buffer=BytesIO(b"%PDF-fake"),
                project_title="Portal Launch",
                audience_type="stakeholder",
            )

        _, _, msg_str = mock_server.sendmail.call_args[0]
        self.assertIn("Stakeholder Report", msg_str)
        self.assertIn("Portal Launch", msg_str)

    @patch('email_sender.smtplib.SMTP')
    def test_subject_reflects_audience_type_engineer(self, mock_smtp_cls):
        """Email subject should say 'Engineering Report' for engineer audience."""
        from email_sender import send_pdf_email
        from unittest.mock import patch, MagicMock
        from io import BytesIO

        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = Mock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = Mock(return_value=False)

        with patch('email_sender.Config') as mock_cfg:
            mock_cfg.SMTP_HOST = "smtp.example.com"
            mock_cfg.SMTP_USER = "sender@example.com"
            mock_cfg.SMTP_PASSWORD = "secret"
            mock_cfg.SMTP_PORT = 587
            mock_cfg.SMTP_USE_TLS = False

            send_pdf_email(
                recipient="engineer@company.com",
                pdf_buffer=BytesIO(b"%PDF-fake"),
                project_title="Infra Upgrade",
                audience_type="engineer",
            )

        _, _, msg_str = mock_server.sendmail.call_args[0]
        self.assertIn("Engineering Report", msg_str)

    @patch('email_sender.smtplib.SMTP')
    def test_pdf_is_attached(self, mock_smtp_cls):
        """PDF bytes should be included in the sent message as an attachment."""
        from email_sender import send_pdf_email
        from unittest.mock import patch, MagicMock
        from io import BytesIO
        import base64

        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = Mock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = Mock(return_value=False)

        fake_pdf = b"%PDF-1.4 fake content"
        with patch('email_sender.Config') as mock_cfg:
            mock_cfg.SMTP_HOST = "smtp.example.com"
            mock_cfg.SMTP_USER = "sender@example.com"
            mock_cfg.SMTP_PASSWORD = "secret"
            mock_cfg.SMTP_PORT = 587
            mock_cfg.SMTP_USE_TLS = False

            send_pdf_email(
                recipient="user@example.com",
                pdf_buffer=BytesIO(fake_pdf),
                project_title="Project",
                audience_type="engineer",
            )

        _, _, msg_str = mock_server.sendmail.call_args[0]
        self.assertIn("application/pdf", msg_str)


class TestFlowForgeStateAudienceType(unittest.TestCase):
    """Tests that FlowForgeState carries audience_type correctly."""

    def test_state_stores_audience_type(self):
        """FlowForgeState should accept and return audience_type."""
        from src.workflow.graph_workflow import FlowForgeState

        state = FlowForgeState(
            proposal="Test",
            prompt="Test",
            hf_token="token",
            audience_type="stakeholder",
        )
        self.assertEqual(state["audience_type"], "stakeholder")
        self.assertEqual(state.audience_type, "stakeholder")

    def test_state_audience_type_defaults_not_present_before_set(self):
        """FlowForgeState raises AttributeError for missing keys, not silent None."""
        from src.workflow.graph_workflow import FlowForgeState

        state = FlowForgeState(proposal="Test", prompt="Test", hf_token="t")
        with self.assertRaises((KeyError, AttributeError)):
            _ = state["audience_type"]

    @patch('src.workflow.graph_workflow.SessionManager')
    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.TimeValidatorAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_run_workflow_sets_audience_type_in_initial_state(
        self,
        mock_validator,
        mock_image,
        mock_plan,
        mock_time_validator,
        mock_time,
        mock_session_manager,
    ):
        """run_flowforge_workflow should pass audience_type into the initial state."""
        from src.workflow.graph_workflow import run_flowforge_workflow

        captured_state = {}

        def capture_execute(state):
            captured_state.update(state)
            return {**state, "timetable": "Phase 1", "milestones": [], "parallel_work_streams": [], "error": None}

        mock_time.return_value.execute = capture_execute
        mock_time_validator.return_value.execute = Mock(return_value={"time_overall_validation": True, "error": None})
        mock_plan.return_value.execute = Mock(return_value={"plan": "plan", "error": None})
        mock_image.return_value.execute = Mock(return_value={"diagrams": [], "diagram_count": 0, "valid_diagram_count": 0, "error": None})
        mock_validator.return_value.execute = Mock(return_value={"validation_results": [], "overall_validation": True, "error": None})

        run_flowforge_workflow(
            proposal="Build a portal",
            prompt="Diagrams please",
            hf_token="test-token",
            audience_type="stakeholder",
            optimize_prompt=False,
        )

        self.assertEqual(captured_state.get("audience_type"), "stakeholder")


if __name__ == '__main__':
    unittest.main()