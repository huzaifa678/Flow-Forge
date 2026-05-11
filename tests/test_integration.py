"""
Integration tests for FlowForge workflow.
Tests the complete agent pipeline from prompt to validation.

"""
import unittest
from unittest.mock import Mock, patch
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from workflow.workflow import create_flowforge_workflow, run_flowforge_workflow


class TestFlowForgeWorkflowIntegration(unittest.TestCase):
    """Integration tests for the complete FlowForge workflow."""
    
    @patch('src.workflow.langgraph_flow.TimeAgent')
    @patch('src.workflow.langgraph_flow.PlanAgent')
    @patch('src.workflow.langgraph_flow.ImageGeneratorAgent')
    @patch('src.workflow.langgraph_flow.ValidatorAgent')
    def test_workflow_execution_success(self, mock_validator, mock_image, mock_plan, mock_time):
        """Test successful end-to-end workflow execution."""
        # Setup mocks to simulate successful execution
        mock_time_instance = Mock()
        mock_time_instance.execute.side_effect = [
            {"timetable": "Phase 1: Design (1 week)\nPhase 2: Development (2 weeks)", "error": None}
        ]
        mock_time.return_value = mock_time_instance
        
        mock_plan_instance = Mock()
        mock_plan_instance.execute.side_effect = [
            {"plan": "Detailed plan: Design tasks, Development tasks, Testing", "error": None}
        ]
        mock_plan.return_value = mock_plan_instance
        
        mock_image_instance = Mock()
        mock_image_instance.execute.side_effect = [
            {"mermaid_diagram": "gantt\ntitle Project\nsection Design\nTask1 :a1, 2026-05-10, 5d\nsection Development\nTask2 :after a1, 10d", "error": None}
        ]
        mock_image.return_value = mock_image_instance
        
        mock_validator_instance = Mock()
        mock_validator_instance.execute.side_effect = [
            {"validation_result": True, "feedback": "Diagram is valid and well-formed", "error": None}
        ]
        mock_validator.return_value = mock_validator_instance
        
        # Run the workflow
        result = run_flowforge_workflow(
            prompt="Create a web application with user authentication",
            hf_token="test-token"
        )
        
        # Verify the result contains expected fields
        self.assertIn("timetable", result)
        self.assertIn("plan", result)
        self.assertIn("mermaid_diagram", result)
        self.assertIn("validation_result", result)
        self.assertIn("feedback", result)
        self.assertIn("current_agent", result)
        
        # Verify final state
        self.assertEqual(result["validation_result"], True)
        self.assertEqual(result["current_agent"], "validator_agent")
        self.assertIsNone(result.get("error"))
        
        # Verify all agents were called
        mock_time_instance.execute.assert_called_once()
        mock_plan_instance.execute.assert_called_once()
        mock_image_instance.execute.assert_called_once()
        mock_validator_instance.execute.assert_called_once()
    
    @patch('src.workflow.langgraph_flow.TimeAgent')
    def test_workflow_execution_time_agent_failure(self, mock_time):
        """Test workflow execution when time agent fails."""
        # Setup mock to simulate failure
        mock_time_instance = Mock()
        mock_time_instance.execute.return_value = {
            "error": "Failed to generate timetable"
        }
        mock_time.return_value = mock_time_instance
        
        # Run the workflow
        result = run_flowforge_workflow(
            prompt="Create a web application",
            hf_token="test-token"
        )
        
        # Verify error is propagated
        self.assertIsNotNone(result.get("error"))
        self.assertIn("Failed to generate timetable", result["error"])
        
        # Verify only time agent was called
        mock_time_instance.execute.assert_called_once()
    
    @patch('src.workflow.langgraph_flow.TimeAgent')
    @patch('src.workflow.langgraph_flow.PlanAgent')
    def test_workflow_execution_plan_agent_failure(self, mock_plan, mock_time):
        """Test workflow execution when plan agent fails."""
        # Setup mocks
        mock_time_instance = Mock()
        mock_time_instance.execute.return_value = {
            "timetable": "Phase 1: Design (1 week)",
            "error": None
        }
        mock_time.return_value = mock_time_instance
        
        mock_plan_instance = Mock()
        mock_plan_instance.execute.return_value = {
            "error": "Failed to generate plan"
        }
        mock_plan.return_value = mock_plan_instance
        
        # Run the workflow
        result = run_flowforge_workflow(
            prompt="Create a web application",
            hf_token="test-token"
        )
        
        # Verify error is propagated
        self.assertIsNotNone(result.get("error"))
        self.assertIn("Failed to generate plan", result["error"])
        
        # Verify time and plan agents were called
        mock_time_instance.execute.assert_called_once()
        mock_plan_instance.execute.assert_called_once()
    
    def test_create_workflow_returns_object(self):
        """Test that create_workflow returns a workflow object."""
        with patch('src.workflow.langgraph_flow.TimeAgent'), \
             patch('src.workflow.langgraph_flow.PlanAgent'), \
             patch('src.workflow.langgraph_flow.ImageGeneratorAgent'), \
             patch('src.workflow.langgraph_flow.ValidatorAgent'):
            
            workflow = create_flowforge_workflow()
            self.assertIsNotNone(workflow)


if __name__ == '__main__':
    unittest.main()