"""Integration tests for FlowForge workflow."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FlowForge'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.workflow.graph_workflow import create_flowforge_workflow, run_flowforge_workflow


class TestFlowForgeWorkflowIntegration(unittest.TestCase):
    """Integration tests for the complete FlowForge workflow."""

    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_workflow_execution_success(self, mock_validator, mock_image, mock_plan, mock_time):
        """Test successful end-to-end workflow execution."""
        # Setup mocks to simulate successful execution
        mock_time_instance = Mock()
        mock_time_instance.execute.side_effect = [
            {
                "timetable": "Phase 1: Design (1 week)\nPhase 2: Development (2 weeks)",
                "milestones": ["Design complete", "Dev complete"],
                "parallel_work_streams": ["Frontend", "Backend"],
                "error": None,
            }
        ]
        mock_time.return_value = mock_time_instance

        mock_plan_instance = Mock()
        mock_plan_instance.execute.side_effect = [
            {
                "plan": "Detailed plan: Design tasks, Development tasks, Testing",
                "error": None,
            }
        ]
        mock_plan.return_value = mock_plan_instance

        mock_image_instance = Mock()
        mock_image_instance.execute.side_effect = [
            {
                "diagrams": [
                    {
                        "diagram_type": "workflow",
                        "mermaid_code": "flowchart TD\n    A[Start] --> B[End]",
                        "is_valid": True,
                        "title": "Workflow",
                    },
                    {
                        "diagram_type": "ci_cd",
                        "mermaid_code": "gantt\ntitle CI/CD Pipeline\nsection Build\nCompile :a1, 2026-05-10, 3d",
                        "is_valid": True,
                        "title": "CI/CD",
                    },
                ],
                "diagram_count": 2,
                "valid_diagram_count": 2,
                "error": None,
            }
        ]
        mock_image.return_value = mock_image_instance

        mock_validator_instance = Mock()
        mock_validator_instance.execute.side_effect = [
            {
                "validation_results": [
                    {"is_valid": True, "diagram_type": "workflow"},
                    {"is_valid": True, "diagram_type": "ci_cd"},
                ],
                "overall_validation": True,
                "valid_count": 2,
                "total_count": 2,
                "error": None,
            }
        ]
        mock_validator.return_value = mock_validator_instance

        # Run the workflow
        result = run_flowforge_workflow(
            proposal="Create a web application with user authentication",
            prompt="Generate workflow and CI/CD diagrams",
            hf_token="test-token",
            optimize_prompt=False,
        )

        # Verify the result contains expected fields
        self.assertIn("timetable", result)
        self.assertIn("plan", result)
        self.assertIn("diagrams", result)
        self.assertIn("validation_results", result)
        self.assertIn("overall_validation", result)
        self.assertIn("current_agent", result)

        # Verify final state
        self.assertEqual(result["overall_validation"], True)
        self.assertEqual(result["current_agent"], "validator_agent")
        self.assertIsNone(result.get("error"))

        # Verify all agents were called
        mock_time_instance.execute.assert_called_once()
        mock_plan_instance.execute.assert_called_once()
        mock_image_instance.execute.assert_called_once()
        mock_validator_instance.execute.assert_called_once()

    @patch('src.workflow.graph_workflow.TimeAgent')
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
            proposal="Create a web application",
            prompt="Generate diagrams",
            hf_token="test-token",
            optimize_prompt=False,
        )

        # Verify error is propagated
        self.assertIsNotNone(result.get("error"))
        self.assertIn("Failed to generate timetable", result["error"])

        # Verify only time agent was called
        mock_time_instance.execute.assert_called_once()

    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    def test_workflow_execution_plan_agent_failure(self, mock_plan, mock_time):
        """Test workflow execution when plan agent fails."""
        # Setup mocks
        mock_time_instance = Mock()
        mock_time_instance.execute.return_value = {
            "timetable": "Phase 1: Design (1 week)",
            "milestones": [],
            "parallel_work_streams": [],
            "error": None,
        }
        mock_time.return_value = mock_time_instance

        mock_plan_instance = Mock()
        mock_plan_instance.execute.return_value = {
            "error": "Failed to generate plan"
        }
        mock_plan.return_value = mock_plan_instance

        # Run the workflow
        result = run_flowforge_workflow(
            proposal="Create a web application",
            prompt="Generate diagrams",
            hf_token="test-token",
            optimize_prompt=False,
        )

        # Verify error is propagated
        self.assertIsNotNone(result.get("error"))
        self.assertIn("Failed to generate plan", result["error"])

        # Verify time and plan agents were called
        mock_time_instance.execute.assert_called_once()
        mock_plan_instance.execute.assert_called_once()

    def test_create_workflow_returns_object(self):
        """Test that create_workflow returns a workflow object."""
        with patch('src.workflow.graph_workflow.TimeAgent'), \
             patch('src.workflow.graph_workflow.PlanAgent'), \
             patch('src.workflow.graph_workflow.ImageGeneratorAgent'), \
             patch('src.workflow.graph_workflow.ValidatorAgent'):

            workflow = create_flowforge_workflow()
            self.assertIsNotNone(workflow)


if __name__ == '__main__':
    unittest.main()