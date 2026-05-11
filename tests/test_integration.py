"""Integration tests for FlowForge workflow."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

from src.schema.helpers import format_workflow_response

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

    def test_format_workflow_response_success(self):
        """Test formatting a successful workflow result."""
        raw_result = {
            "proposal_summary": "Test project",
            "optimized_prompt": "Optimized prompt",
            "timetable": "gantt\ntitle Test\nsection A\nTask :a1, 2026-05-10, 5d",
            "milestones": ["Phase 1", "Phase 2"],
            "parallel_work_streams": ["Frontend", "Backend"],
            "plan": "Detailed plan",
            "diagrams": [
                {
                    "diagram_type": "workflow",
                    "mermaid_code": "flowchart TD\n    A --> B",
                    "is_valid": True,
                    "title": "Workflow",
                }
            ],
            "diagram_count": 1,
            "valid_diagram_count": 1,
            "validation_results": [{"is_valid": True}],
            "overall_validation": True,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = format_workflow_response(raw_result)

        self.assertEqual(response.status, "success")
        self.assertEqual(response.valid_diagram_count, 1)
        self.assertEqual(response.total_diagram_count, 1)
        self.assertTrue(response.overall_validation)

    def test_format_workflow_response_partial(self):
        """Test formatting a partial workflow result (some diagrams invalid)."""
        raw_result = {
            "proposal_summary": "Test project",
            "diagrams": [
                {"diagram_type": "workflow", "is_valid": True},
                {"diagram_type": "ci_cd", "is_valid": False},
            ],
            "diagram_count": 2,
            "valid_diagram_count": 1,
            "overall_validation": False,
            "error": None,
        }

        response = format_workflow_response(raw_result)
        self.assertEqual(response.status, "partial")

    def test_format_workflow_response_failed(self):
        """Test formatting a failed workflow result."""
        raw_result = {
            "proposal_summary": "Test project",
            "diagrams": [],
            "diagram_count": 0,
            "valid_diagram_count": 0,
            "error": "Something went wrong",
        }

        response = format_workflow_response(raw_result)
        self.assertEqual(response.status, "failed")
        self.assertEqual(response.error, "Something went wrong")


class TestPromptOptimization(unittest.TestCase):
    """Integration tests for prompt optimization in workflow."""

    @patch('src.workflow.graph_workflow.PromptOptimizer')
    @patch('src.workflow.graph_workflow.TimeAgent')
    def test_workflow_with_optimization(self, mock_time, mock_optimizer_cls):
        """Test that workflow applies prompt optimization when enabled."""
        mock_optimizer_instance = Mock()
        mock_optimizer_instance.optimize.return_value = {
            "original_prompt": "Create a web app",
            "optimized_prompt": "Optimized: Create a web application with...",
            "optimization_technique": "langchain_llm_router",
        }
        mock_optimizer_cls.return_value = mock_optimizer_instance

        mock_time_instance = Mock()
        mock_time_instance.execute.return_value = {
            "timetable": "Phase 1: Design",
            "milestones": [],
            "parallel_work_streams": [],
            "error": None,
        }
        mock_time.return_value = mock_time_instance

        from src.workflow.graph_workflow import run_flowforge_workflow

        result = run_flowforge_workflow(
            proposal="Create a web app",
            prompt="Create a web app",
            hf_token="test-token",
            optimize_prompt=True,
        )

        # Verify optimization was called
        mock_optimizer_instance.optimize.assert_called_once_with("Create a web app")

    @patch('src.workflow.graph_workflow.TimeAgent')
    def test_workflow_without_optimization(self, mock_time):
        """Test that workflow skips optimization when disabled."""
        mock_time_instance = Mock()
        mock_time_instance.execute.return_value = {
            "timetable": "Phase 1: Design",
            "milestones": [],
            "parallel_work_streams": [],
            "error": None,
        }
        mock_time.return_value = mock_time_instance

        from src.workflow.graph_workflow import run_flowforge_workflow

        result = run_flowforge_workflow(
            proposal="Create a web app",
            prompt="Create a web app",
            hf_token="test-token",
            optimize_prompt=False,
        )

        # Should still work without optimization
        self.assertIn("timetable", result)


if __name__ == '__main__':
    unittest.main()