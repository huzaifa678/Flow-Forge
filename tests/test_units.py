"""Tests for FlowForge agents and workflow."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

from src.schemas.request import PromptRequest
from src.pipeline.prompt_optimizer import PromptOptimizer

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FlowForge'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.time_agent import TimeAgent
from src.agents.plan_agent import PlanAgent
from src.agents.image_generator_agent import ImageGeneratorAgent
from src.agents.validator_agent import ValidatorAgent
from src.workflow.graph_workflow import create_flowforge_workflow
from src.schemas.request import (
    DiagramType,
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


class TestValidatorAgent(unittest.TestCase):
    """Test cases for ValidatorAgent - now validates all diagrams."""

    @patch('src.agents.validator_agent.InferenceClient')
    def test_initialization(self, mock_llm):
        """Test ValidatorAgent initialization."""
        agent = ValidatorAgent()
        self.assertEqual(agent.name, "validator_agent")
        mock_llm.assert_called_once()

    @patch('src.agents.validator_agent.InferenceClient')
    def test_execute_success_valid(self, mock_llm):
        """Test successful execution of ValidatorAgent with valid diagram."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""VALID: true
CRITICAL_ISSUES: None
WARNINGS: None
SUGGESTIONS: None
FEEDBACK: Diagram is well-formed and follows best practices."""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ValidatorAgent()
        state = {
            "diagrams": [
                {
                    "mermaid_code": "gantt\ntitle Test Chart\nsection A\nTask :a1, 2026-05-10, 5d",
                    "diagram_type": "workflow",
                }
            ],
        }

        result = agent.execute(state)

        self.assertIn("validation_results", result)
        self.assertIn("overall_validation", result)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["current_agent"], "validator_agent")

    @patch('src.agents.validator_agent.InferenceClient')
    def test_execute_empty_diagrams(self, mock_llm):
        """Test ValidatorAgent with empty diagrams list."""
        agent = ValidatorAgent()
        state = {"diagrams": []}

        result = agent.execute(state)

        self.assertIn("error", result)
        self.assertIn("No diagrams", result["error"])

    def test_basic_validation_passes(self):
        """Test basic Mermaid validation passes for valid diagrams."""
        agent = ValidatorAgent()
        result = agent._basic_mermaid_validation(
            "gantt\ntitle Test\nsection A\nTask :a1, 2026-05-10, 5d"
        )
        self.assertTrue(result["is_valid"])

    def test_basic_validation_fails_empty(self):
        """Test basic Mermaid validation fails for empty input."""
        agent = ValidatorAgent()
        result = agent._basic_mermaid_validation("")
        self.assertFalse(result["is_valid"])

    def test_basic_validation_fails_no_mermaid(self):
        """Test basic Mermaid validation fails for non-Mermaid text."""
        agent = ValidatorAgent()
        result = agent._basic_mermaid_validation("This is just text")
        self.assertFalse(result["is_valid"])

    def test_basic_validation_fails_brackets(self):
        """Test basic Mermaid validation fails for mismatched brackets."""
        agent = ValidatorAgent()
        result = agent._basic_mermaid_validation("gantt\ntitle [Unclosed")
        self.assertFalse(result["is_valid"])

    @patch('src.agents.validator_agent.InferenceClient')
    def test_parse_validation_response(self, mock_llm):
        """Test parsing of LLM validation response."""
        agent = ValidatorAgent()
        response = """VALID: true
CRITICAL_ISSUES: None
WARNINGS: Minor suggestion about naming
SUGGESTIONS: Use more descriptive node names
FEEDBACK: Overall the diagram is well-formed and logically consistent."""
        result = agent._parse_llm_validation(response)
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["feedback"], "Overall the diagram is well-formed and logically consistent.")

    @patch('src.agents.validator_agent.InferenceClient')
    def test_parse_validation_response_invalid(self, mock_llm):
        """Test parsing of invalid LLM validation response."""
        agent = ValidatorAgent()
        response = """VALID: false
CRITICAL_ISSUES: Circular dependency detected
WARNINGS: Node naming could be improved
SUGGESTIONS: None
FEEDBACK: The diagram has a circular reference."""
        result = agent._parse_llm_validation(response)
        self.assertFalse(result["is_valid"])
        self.assertIn("Circular dependency", result["critical_issues"][0])

    @patch('src.agents.validator_agent.InferenceClient')
    def test_legacy_diagram_support(self, mock_llm):
        """Test that validator supports legacy 'mermaid_diagram' field."""
        mock_instance = Mock()
        mock_instance.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="""VALID: true
CRITICAL_ISSUES: None
WARNINGS: None
SUGGESTIONS: None
FEEDBACK: Diagram is valid."""
            ))]
        )
        mock_llm.return_value = mock_instance

        agent = ValidatorAgent()
        # Test with legacy single-diagram state
        state = {
            "mermaid_diagram": "gantt\ntitle Test\nsection A\nTask :a1, 2026-05-10, 5d",
        }

        result = agent.execute(state)
        self.assertIn("validation_results", result)
        self.assertEqual(len(result["validation_results"]), 1)


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


class TestWorkflow(unittest.TestCase):
    """Test cases for the workflow."""

    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_create_workflow(self, mock_validator, mock_image, mock_plan, mock_time):
        """Test workflow creation."""
        mock_time.return_value.execute = Mock(return_value={"timetable": "test"})
        mock_plan.return_value.execute = Mock(return_value={"plan": "test"})
        mock_image.return_value.execute = Mock(return_value={"diagrams": []})
        mock_validator.return_value.execute = Mock(return_value={"validation_results": []})

        workflow = create_flowforge_workflow()

        self.assertIsNotNone(workflow)
        mock_time.assert_called_once()
        mock_plan.assert_called_once()
        mock_image.assert_called_once()
        mock_validator.assert_called_once()

    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_workflow_state_flow(
        self,
        mock_validator,
        mock_image,
        mock_plan,
        mock_time,
    ):
        """Test that workflow state flows correctly through agents."""

        # -------------------------
        # Time Agent mock
        # -------------------------
        time_instance = Mock()
        time_instance.execute.return_value = {
            "timetable": "Phase 1: Design, Phase 2: Dev",
            "milestones": ["Design complete", "Development complete"],
            "parallel_work_streams": ["Frontend", "Backend"],
            "error": None,
        }
        mock_time.return_value = time_instance

        plan_instance = Mock()
        plan_instance.execute.return_value = {
            "plan": "Task breakdown with resources",
            "error": None,
        }
        mock_plan.return_value = plan_instance

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

        # -------------------------
        # Assertions
        # -------------------------
        self.assertIsNotNone(result)
        self.assertIn("diagrams", result)
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


if __name__ == '__main__':
    unittest.main()