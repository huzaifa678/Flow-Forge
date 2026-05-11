"""Tests for FlowForge agents and workflow."""
import unittest
from unittest.mock import Mock, patch
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.time_agent import TimeAgent
from src.agents.plan_agent import PlanAgent
from src.agents.image_generator_agent import ImageGeneratorAgent
from src.agents.validator_agent import ValidatorAgent
from src.workflow.graph_workflow import create_flowforge_workflow, run_flowforge_workflow


class TestBaseAgent(unittest.TestCase):
    """Test cases for base agent functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass


class TestTimeAgent(unittest.TestCase):
    """Test cases for TimeAgent."""
    
    @patch('src.agents.time_agent.HuggingFaceEndpoint')
    def test_initialization(self, mock_llm):
        """Test TimeAgent initialization."""
        agent = TimeAgent()
        self.assertEqual(agent.name, "time_agent")
        mock_llm.assert_called_once()
    
    @patch('src.agents.time_agent.HuggingFaceEndpoint')
    def test_execute_success(self, mock_llm):
        """Test successful execution of TimeAgent."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = "Timetable: Phase 1 (2 weeks), Phase 2 (3 weeks)"
        mock_llm.return_value = mock_instance
        
        agent = TimeAgent()
        state = {
            "prompt": "Create a website",
            "hf_token": "test-token"
        }
        
        result = agent.execute(state)
        
        self.assertIn("timetable", result)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["current_agent"], "time_agent")
    
    @patch('src.agents.time_agent.HuggingFaceEndpoint')
    def test_execute_no_prompt(self, mock_llm):
        """Test TimeAgent execution with no prompt."""
        agent = TimeAgent()
        state = {
            "prompt": "",
            "hf_token": "test-token"
        }
        
        result = agent.execute(state)
        
        self.assertIsNotNone(result.get("error"))
        self.assertIn("No prompt provided", result["error"])


class TestPlanAgent(unittest.TestCase):
    """Test cases for PlanAgent."""
    
    @patch('src.agents.plan_agent.HuggingFaceEndpoint')
    def test_initialization(self, mock_llm):
        """Test PlanAgent initialization."""
        agent = PlanAgent()
        self.assertEqual(agent.name, "plan_agent")
        mock_llm.assert_called_once()
    
    @patch('src.agents.plan_agent.HuggingFaceEndpoint')
    def test_execute_success(self, mock_llm):
        """Test successful execution of PlanAgent."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = "Detailed plan with tasks and resources"
        mock_llm.return_value = mock_instance
        
        agent = PlanAgent()
        state = {
            "timetable": "Phase 1: Design (1 week), Phase 2: Development (2 weeks)",
            "hf_token": "test-token"
        }
        
        result = agent.execute(state)
        
        self.assertIn("plan", result)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["current_agent"], "plan_agent")


class TestImageGeneratorAgent(unittest.TestCase):
    """Test cases for ImageGeneratorAgent."""
    
    @patch('src.agents.image_generator_agent.HuggingFaceEndpoint')
    def test_initialization(self, mock_llm):
        """Test ImageGeneratorAgent initialization."""
        agent = ImageGeneratorAgent()
        self.assertEqual(agent.name, "image_agent")
        mock_llm.assert_called_once()
    
    @patch('src.agents.image_generator_agent.HuggingFaceEndpoint')
    def test_execute_success(self, mock_llm):
        """Test successful execution of ImageGeneratorAgent."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = "gantt\n    title Test Chart\n    section Section\n    Task :a1, 2026-05-10, 10d"
        mock_llm.return_value = mock_instance
        
        agent = ImageGeneratorAgent()
        state = {
            "plan": "Plan for testing",
            "hf_token": "test-token"
        }
        
        result = agent.execute(state)
        
        self.assertIn("mermaid_diagram", result)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["current_agent"], "image_agent")


class TestValidatorAgent(unittest.TestCase):
    """Test cases for ValidatorAgent."""
    
    @patch('src.agents.validator_agent.HuggingFaceEndpoint')
    def test_initialization(self, mock_llm):
        """Test ValidatorAgent initialization."""
        agent = ValidatorAgent()
        self.assertEqual(agent.name, "validator_agent")
        mock_llm.assert_called_once()
    
    @patch('src.agents.validator_agent.HuggingFaceEndpoint')
    def test_execute_success_valid(self, mock_llm):
        """Test successful execution of ValidatorAgent with valid diagram."""
        # Setup mock
        mock_instance = Mock()
        mock_instance.invoke.return_value = "VALID: true\nFEEDBACK: Diagram is well-formed and logical"
        mock_llm.return_value = mock_instance
        
        agent = ValidatorAgent()
        state = {
            "mermaid_diagram": "gantt\n    title Test Chart\n    section Section\n    Task :a1, 2026-05-10, 10d",
            "hf_token": "test-token"
        }
        
        result = agent.execute(state)
        
        self.assertIn("validation_result", result)
        self.assertIn("feedback", result)
        self.assertIsNone(result.get("error"))
        self.assertEqual(result["current_agent"], "validator_agent")
        self.assertTrue(result["validation_result"])


class TestWorkflow(unittest.TestCase):
    """Test cases for the workflow."""
    
    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_create_workflow(self, mock_validator, mock_image, mock_plan, mock_time):
        """Test workflow creation."""
        # Setup mocks
        mock_time.return_value.execute = Mock(return_value={"timetable": "test"})
        mock_plan.return_value.execute = Mock(return_value={"plan": "test"})
        mock_image.return_value.execute = Mock(return_value={"mermaid_diagram": "test"})
        mock_validator.return_value.execute = Mock(return_value={"validation_result": True})
        
        workflow = create_flowforge_workflow()
        
        self.assertIsNotNone(workflow)
        # Check that agents were instantiated
        mock_time.assert_called_once()
        mock_plan.assert_called_once()
        mock_image.assert_called_once()
        mock_validator.assert_called_once()


if __name__ == '__main__':
    unittest.main()