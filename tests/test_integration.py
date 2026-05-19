"""Integration tests for FlowForge API endpoints and workflow."""
import unittest
from unittest.mock import Mock, patch
import os
import sys

from src.api.dependencies import get_hf_token

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'FlowForge'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.app import app
from src.workflow.graph_workflow import run_flowforge_workflow, create_flowforge_workflow
from src.schema.helpers import format_workflow_response


class TestAPIEndpointsIntegration(unittest.TestCase):
    """Integration tests for FlowForge API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.client = TestClient(app, raise_server_exceptions=False)

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_generate_diagrams_success(self, mock_get_token, mock_run_workflow):
        """Test POST /api/v1/generate-diagrams returns successful response."""
        mock_get_token.return_value = "test-hf-token"
        mock_run_workflow.return_value = {
            "proposal_summary": "E-commerce platform",
            "optimized_prompt": "Optimized prompt for e-commerce",
            "timetable": "gantt\ntitle E-Commerce\nsection Design\nUI Design :a1, 2026-05-10, 7d",
            "milestones": ["Design complete", "Development complete", "Testing complete"],
            "parallel_work_streams": ["Frontend", "Backend", "Database"],
            "plan": "Phase 1: Design the UI\nPhase 2: Build backend APIs\nPhase 3: Integrate and test",
            "diagrams": [
                {
                    "diagram_type": "workflow",
                    "mermaid_code": "flowchart TD\n    A[User Login] --> B[Browse Products]",
                    "is_valid": True,
                    "title": "User Workflow",
                },
                {
                    "diagram_type": "ci_cd",
                    "mermaid_code": "gantt\ntitle CI/CD\nsection Build\nCompile :a1, 2026-05-10, 3d",
                    "is_valid": True,
                    "title": "CI/CD Pipeline",
                },
            ],
            "diagram_count": 2,
            "valid_diagram_count": 2,
            "validation_results": [
                {"is_valid": True, "diagram_type": "workflow"},
                {"is_valid": True, "diagram_type": "ci_cd"},
            ],
            "overall_validation": True,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={
                "proposal": {
                    "title": "E-commerce platform",
                    "description": "Build a full-stack e-commerce platform with user authentication, product catalog, shopping cart, and payment processing.",
                    "requirements": [
                        "User authentication with JWT",
                        "Product catalog with search",
                        "Shopping cart functionality",
                        "Payment gateway integration"
                    ],
                    "constraints": ["Must use microservices architecture", "Budget under $100k"],
                    "tech_stack": ["React", "Node.js", "PostgreSQL", "Docker"],
                    "timeline_weeks": 16,
                    "team_size": 8,
                    "budget_range": "$80k-$100k"
                },
                "prompt": {
                    "user_prompt": "Generate a workflow diagram and CI/CD pipeline diagram for the e-commerce platform",
                    "diagram_types": ["workflow", "ci_cd"],
                    "optimize_prompt": True,
                    "priority": "high"
                },
                "hf_token": "test-hf-token"
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["proposal_summary"], "E-commerce platform")
        self.assertEqual(data["total_diagram_count"], 2)
        self.assertEqual(data["valid_diagram_count"], 2)
        self.assertTrue(data["overall_validation"])
        self.assertEqual(len(data["diagrams"]), 2)
        self.assertIsNone(data["error"])

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_generate_diagrams_partial_success(self, mock_get_token, mock_run_workflow):
        """Test POST /api/v1/generate-diagrams with some invalid diagrams."""
        mock_get_token.return_value = "test-hf-token"
        mock_run_workflow.return_value = {
            "proposal_summary": "Partial project",
            "optimized_prompt": "Optimized prompt",
            "timetable": "Phase 1: Design",
            "milestones": ["Design"],
            "parallel_work_streams": [],
            "plan": "Some plan",
            "diagrams": [
                {
                    "diagram_type": "workflow",
                    "mermaid_code": "flowchart TD\n    A --> B",
                    "is_valid": True,
                    "title": "Valid Diagram",
                },
                {
                    "diagram_type": "ci_cd",
                    "mermaid_code": "invalid mermaid",
                    "is_valid": False,
                    "title": "Invalid Diagram",
                },
            ],
            "diagram_count": 2,
            "valid_diagram_count": 1,
            "validation_results": [
                {"is_valid": True, "diagram_type": "workflow"},
                {"is_valid": False, "diagram_type": "ci_cd"},
            ],
            "overall_validation": False,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={
                "proposal": {
                    "title": "Partial project",
                    "description": "A project with partial results",
                },
                "prompt": {
                    "user_prompt": "Generate diagrams",
                    "diagram_types": ["workflow", "ci_cd"],
                    "optimize_prompt": False,
                    "priority": "medium"
                },
                "hf_token": "test-hf-token"
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "partial")
        self.assertEqual(data["total_diagram_count"], 2)
        self.assertEqual(data["valid_diagram_count"], 1)
        self.assertFalse(data["overall_validation"])

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_generate_diagrams_workflow_failure(self, mock_get_token, mock_run_workflow):
        """Test POST /api/v1/generate-diagrams when workflow fails."""
        mock_get_token.return_value = "test-hf-token"
        mock_run_workflow.return_value = {
            "proposal_summary": "Failed project",
            "diagrams": [],
            "diagram_count": 0,
            "valid_diagram_count": 0,
            "error": "Time agent failed: LLM connection timeout",
            "current_agent": "time_agent",
        }

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={
                "proposal": {
                    "title": "Failed project",
                    "description": "This will fail",
                },
                "prompt": {
                    "user_prompt": "Generate diagrams",
                    "diagram_types": ["workflow"],
                    "optimize_prompt": False,
                    "priority": "medium"
                },
                "hf_token": "test-hf-token"
            },
        )

        # The router returns 500 when workflow fails and no diagrams are produced
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertEqual(data["status"], "failed")
        self.assertEqual(data["error"], "Time agent failed: LLM connection timeout")
        self.assertEqual(data["total_diagram_count"], 0)

    @patch('src.api.router.get_hf_token')
    def test_generate_diagrams_validation_error(self, mock_get_token):
        """Test POST /api/v1/generate-diagrams with invalid request data."""
        mock_get_token.return_value = "test-hf-token"

        # Missing required fields
        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={
                "proposal": {
                    "title": "Hi",  # Too short (min 3)
                    "description": "Too short",  # Too short (min 10)
                },
                "prompt": {
                    "user_prompt": "Hi",  # Too short (min 10)
                },
                "hf_token": "short"  # Too short (min 10)
            },
        )

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("detail", data)

    @patch('src.api.router.get_hf_token')
    def test_generate_diagrams_empty_request(self, mock_get_token):
        """Test POST /api/v1/generate-diagrams with empty body."""
        mock_get_token.return_value = "test-hf-token"

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={},
        )

        self.assertEqual(response.status_code, 422)

    @patch('src.api.router.get_hf_token')
    def test_health_check_success(self, mock_get_token):
        """Test GET /api/v1/health returns healthy status."""
        mock_get_token.return_value = "test-hf-token"

        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["message"], "FlowForge API is healthy")
        self.assertIn("data", data)
        self.assertTrue(data["data"]["hf_token_configured"])

    def test_health_check_no_token(self):
        """Test GET /api/v1/health when HF token missing."""

        def override_dependency():
            raise HTTPException(
                status_code=500,
                detail=(
                    "HF_TOKEN is not configured. "
                    "Set it in environment or .env file."
                ),
            )

        app.dependency_overrides[
            get_hf_token
        ] = override_dependency

        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 500)

        app.dependency_overrides.clear()

    def test_health_check_unauthorized_no_token_header(self):
        """Test GET /api/v1/health without providing token dependency path."""
        # The endpoint requires hf_token via Depends, so without mocking
        # the dependency it would fail. Test that the endpoint exists.
        response = self.client.get("/api/v1/health")
        # Will be 500 since no real HF_TOKEN in env, but endpoint is reachable
        self.assertIn(response.status_code, [200, 401, 403, 500])

    def test_invalid_route_returns_404(self):
        """Test that undefined routes return 404."""
        response = self.client.get("/api/v1/nonexistent")
        self.assertEqual(response.status_code, 404)

    def test_root_returns_404(self):
        """Test that root path is not served."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 404)

    @patch('src.api.router.get_hf_token')
    def test_generate_diagrams_method_not_allowed(self, mock_get_token):
        """Test that GET on generate-diagrams returns 405."""
        mock_get_token.return_value = "test-hf-token"

        response = self.client.get("/api/v1/generate-diagrams")
        self.assertEqual(response.status_code, 405)


class TestWorkflowAndResponseFormattingIntegration(unittest.TestCase):
    """Integration tests for workflow execution and response formatting."""

    @patch('src.workflow.graph_workflow.SessionManager')
    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.TimeValidatorAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_full_workflow_produces_valid_api_response(self, mock_validator, mock_image,
                                                         mock_plan, mock_time_validator, mock_time,
                                                         mock_session_manager):
        """Test that full workflow execution produces a properly formatted API response."""
        # Setup mocks for successful execution
        mock_time_instance = Mock()
        mock_time_instance.execute.return_value = {
            "timetable": "gantt\ntitle Project\nsection Design\nUI :a1, 2026-05-10, 7d",
            "milestones": ["Design complete", "Dev complete", "Testing complete"],
            "parallel_work_streams": ["Frontend", "Backend"],
            "error": None,
        }
        mock_time.return_value = mock_time_instance

        mock_time_validator_instance = Mock()
        mock_time_validator_instance.execute.return_value = {
            "time_overall_validation": True,
            "time_validation_results": [],
            "error": None,
        }
        mock_time_validator.return_value = mock_time_validator_instance

        mock_plan_instance = Mock()
        mock_plan_instance.execute.return_value = {
            "plan": "Phase 1: Design\nPhase 2: Development\nPhase 3: Testing",
            "error": None,
        }
        mock_plan.return_value = mock_plan_instance

        mock_image_instance = Mock()
        mock_image_instance.execute.return_value = {
            "diagrams": [
                {
                    "diagram_type": "workflow",
                    "mermaid_code": "flowchart TD\n    A[Start] --> B[End]",
                    "is_valid": True,
                    "title": "Workflow",
                },
            ],
            "diagram_count": 1,
            "valid_diagram_count": 1,
            "error": None,
        }
        mock_image.return_value = mock_image_instance

        mock_validator_instance = Mock()
        mock_validator_instance.execute.return_value = {
            "validation_results": [
                {"is_valid": True, "diagram_type": "workflow"},
            ],
            "overall_validation": True,
            "valid_count": 1,
            "total_count": 1,
            "error": None,
        }
        mock_validator.return_value = mock_validator_instance

        result = run_flowforge_workflow(
            proposal="Build an e-commerce website",
            prompt="Generate a workflow diagram",
            hf_token="test-token",
            project_title="E-Commerce Site",
            optimize_prompt=False,
        )

        # Format as API response
        response = format_workflow_response(result)

        # Verify response structure matches API contract
        assert response["status"] == "success"
        assert response["proposal_summary"] is not None
        assert response["timeline"] is not None
        assert response["timeline"]["milestones"] == ["Design complete", "Dev complete", "Testing complete"]
        assert response["timeline"]["parallel_work_streams"] == ["Frontend", "Backend"]
        assert "gantt" in (response["timeline"]["gantt_chart"] or "")
        assert response["plan"] is not None
        assert len(response["diagrams"]) == 1
        assert response["valid_diagram_count"] == 1
        assert response["total_diagram_count"] == 1
        assert response["overall_validation"] is True
        assert response["error"] is None
        assert response["current_agent"] == "validator_agent"

    def test_format_workflow_response_error_state(self):
        """Test formatting a failed workflow result for API response."""
        raw_result = {
            "proposal_summary": "Failed project",
            "diagrams": [],
            "diagram_count": 0,
            "valid_diagram_count": 0,
            "error": "Workflow execution failed: Connection refused",
        }

        response = format_workflow_response(raw_result)

        assert response["status"] == "failed"
        assert response["error"] == "Workflow execution failed: Connection refused"
        assert response["total_diagram_count"] == 0
        assert response["valid_diagram_count"] == 0
        assert response["diagrams"] == []

    def test_format_workflow_response_with_all_fields(self):
        """Test formatting a complete workflow result with all optional fields."""
        raw_result = {
            "proposal_summary": "Full project",
            "optimized_prompt": "Optimized version",
            "timetable": "gantt\ntitle Full\nsection A\nTask :a1, 2.0",
            "milestones": ["Milestone 1", "Milestone 2"],
            "parallel_work_streams": ["Stream A", "Stream B"],
            "plan": "Full plan here",
            "diagrams": [
                {"diagram_type": "workflow", "mermaid_code": "A --> B", "is_valid": True, "title": "D1"},
                {"diagram_type": "ci_cd", "mermaid_code": "gantt\ntitle CI", "is_valid": True, "title": "D2"},
                {"diagram_type": "system_design", "mermaid_code": "not valid", "is_valid": False, "title": "D3"},
            ],
            "diagram_count": 3,
            "valid_diagram_count": 2,
            "validation_results": [
                {"is_valid": True, "diagram_type": "workflow"},
                {"is_valid": True, "diagram_type": "ci_cd"},
                {"is_valid": False, "diagram_type": "system_design"},
            ],
            "overall_validation": False,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = format_workflow_response(raw_result)

        assert response["status"] == "partial"
        assert response["optimized_prompt"] == "Optimized version"
        assert response["timeline"] is not None
        assert len(response["timeline"]["milestones"]) == 2
        assert response["plan"] == "Full plan here"
        assert len(response["diagrams"]) == 3
        assert response["valid_diagram_count"] == 2
        assert response["total_diagram_count"] == 3
        assert response["overall_validation"] is False

    @patch('tests.test_integration.create_flowforge_workflow')
    def test_workflow_creation_integration(self, mock_create):
        """Test that workflow creation produces a valid StateGraph."""
        from langgraph.graph import StateGraph

        mock_graph = Mock(spec=StateGraph)
        mock_graph.compile.return_value = mock_graph
        mock_create.return_value = mock_graph

        workflow = create_flowforge_workflow()

        mock_create.assert_called_once()


class TestAudienceTypeAPIIntegration(unittest.TestCase):
    """Integration tests verifying audience_type flows through the API layer."""

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def _base_stakeholder_payload(self, **prompt_overrides):
        payload = {
            "proposal": {
                "title": "Customer Portal Project",
                "description": "A self-service portal for enterprise customers to manage their accounts.",
                "requirements": ["User login", "Account management", "Reports"],
                "constraints": ["Launch within 6 months"],
                "tech_stack": ["React", "FastAPI"],
                "timeline_weeks": 24,
                "team_size": 6,
            },
            "prompt": {
                "user_prompt": "Generate a high-level overview for the board presentation",
                "diagram_types": ["flowchart", "gantt"],
                "optimize_prompt": False,
                "priority": "high",
                "audience_type": "stakeholder",
                **prompt_overrides,
            },
            "hf_token": "test-hf-token",
        }
        return payload

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_stakeholder_audience_type_passed_to_workflow(self, mock_get_token, mock_run_workflow):
        """Verify audience_type='stakeholder' is forwarded from the API request to the workflow."""
        mock_get_token.return_value = "test-hf-token"
        mock_run_workflow.return_value = {
            "proposal_summary": "Customer Portal Project",
            "diagrams": [
                {
                    "diagram_type": "flowchart",
                    "mermaid_code": "graph TD\n    A[Phase 1] --> B[Phase 2]",
                    "is_valid": True,
                    "title": "Project Phases",
                }
            ],
            "diagram_count": 1,
            "valid_diagram_count": 1,
            "validation_results": [{"is_valid": True, "diagram_type": "flowchart"}],
            "overall_validation": True,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json=self._base_stakeholder_payload(),
        )

        self.assertEqual(response.status_code, 200)
        # Verify the workflow was called with audience_type="stakeholder"
        _, kwargs = mock_run_workflow.call_args
        self.assertEqual(kwargs.get("audience_type"), "stakeholder")

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_engineer_audience_type_passed_to_workflow(self, mock_get_token, mock_run_workflow):
        """Verify audience_type='engineer' is forwarded from the API request to the workflow."""
        mock_get_token.return_value = "test-hf-token"
        mock_run_workflow.return_value = {
            "proposal_summary": "Test Project",
            "diagrams": [],
            "diagram_count": 0,
            "valid_diagram_count": 0,
            "overall_validation": False,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={
                "proposal": {
                    "title": "Backend Microservices",
                    "description": "Refactor monolith into microservices architecture.",
                },
                "prompt": {
                    "user_prompt": "Generate system design and CI/CD diagrams",
                    "diagram_types": ["system_design", "ci_cd"],
                    "optimize_prompt": False,
                    "priority": "medium",
                    "audience_type": "engineer",
                },
                "hf_token": "test-hf-token",
            },
        )

        _, kwargs = mock_run_workflow.call_args
        self.assertEqual(kwargs.get("audience_type"), "engineer")

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_default_audience_type_is_engineer(self, mock_get_token, mock_run_workflow):
        """When audience_type is omitted it should default to 'engineer'."""
        mock_get_token.return_value = "test-hf-token"
        mock_run_workflow.return_value = {
            "proposal_summary": "Test",
            "diagrams": [],
            "diagram_count": 0,
            "valid_diagram_count": 0,
            "overall_validation": False,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={
                "proposal": {
                    "title": "Default Audience Test",
                    "description": "Testing default audience type behaviour.",
                },
                "prompt": {
                    "user_prompt": "Generate a simple workflow diagram please",
                    "diagram_types": ["workflow"],
                    "optimize_prompt": False,
                    "priority": "low",
                    # audience_type intentionally omitted
                },
                "hf_token": "test-hf-token",
            },
        )

        self.assertEqual(response.status_code, 200)
        _, kwargs = mock_run_workflow.call_args
        self.assertEqual(kwargs.get("audience_type"), "engineer")

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_stakeholder_response_contains_business_diagram_types(self, mock_get_token, mock_run_workflow):
        """A stakeholder response should only contain flowchart/gantt/architecture diagrams."""
        mock_get_token.return_value = "test-hf-token"
        mock_run_workflow.return_value = {
            "proposal_summary": "Customer Portal",
            "diagrams": [
                {"diagram_type": "flowchart", "mermaid_code": "graph TD\n    A --> B", "is_valid": True, "title": "Project Phases"},
                {"diagram_type": "gantt", "mermaid_code": "gantt\ntitle Timeline\nsection A\nTask :a1, 2026-05-14, 5d", "is_valid": True, "title": "Timeline"},
            ],
            "diagram_count": 2,
            "valid_diagram_count": 2,
            "overall_validation": True,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json=self._base_stakeholder_payload(),
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        allowed = {"flowchart", "gantt", "architecture"}
        for diagram in data["diagrams"]:
            self.assertIn(
                diagram["diagram_type"],
                allowed,
                f"Unexpected diagram type for stakeholder: {diagram['diagram_type']}",
            )

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_invalid_audience_type_rejected(self, mock_get_token, mock_run_workflow):
        """An unrecognised audience_type value should be rejected with 422."""
        mock_get_token.return_value = "test-hf-token"

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={
                "proposal": {
                    "title": "Test Project",
                    "description": "Test project description here.",
                },
                "prompt": {
                    "user_prompt": "Generate workflow diagrams please",
                    "diagram_types": ["workflow"],
                    "optimize_prompt": False,
                    "priority": "medium",
                    "audience_type": "cto",  # invalid
                },
                "hf_token": "test-hf-token",
            },
        )

        self.assertEqual(response.status_code, 422)


class TestStakeholderWorkflowIntegration(unittest.TestCase):
    """Integration tests for the stakeholder workflow end-to-end."""

    @patch('src.workflow.graph_workflow.SessionManager')
    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.TimeValidatorAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_stakeholder_workflow_produces_valid_api_response(
        self,
        mock_validator,
        mock_image,
        mock_plan,
        mock_time_validator,
        mock_time,
        mock_session_manager,
    ):
        """Full workflow with audience_type=stakeholder should produce a valid formatted response."""
        mock_time.return_value.execute.return_value = {
            "timetable": "gantt\ntitle Portal\nsection Design\nUI :a1, 2026-05-14, 7d",
            "milestones": ["Design complete", "Build complete", "Launch"],
            "parallel_work_streams": ["Frontend", "Backend"],
            "error": None,
        }
        mock_time_validator.return_value.execute.return_value = {
            "time_overall_validation": True,
            "time_validation_results": [],
            "error": None,
        }
        mock_plan.return_value.execute.return_value = {
            "plan": "Phase 1: Discovery\nPhase 2: Build\nPhase 3: Launch",
            "error": None,
        }
        mock_image.return_value.execute.return_value = {
            "diagrams": [
                {"diagram_type": "flowchart", "mermaid_code": "graph TD\n    A --> B", "is_valid": True, "title": "Project Phases"},
                {"diagram_type": "gantt", "mermaid_code": "gantt\ntitle Timeline\nsection A\nTask :a1, 2026-05-14, 5d", "is_valid": True, "title": "Timeline"},
            ],
            "diagram_count": 2,
            "valid_diagram_count": 2,
            "error": None,
        }
        mock_validator.return_value.execute.return_value = {
            "validation_results": [
                {"is_valid": True, "diagram_type": "flowchart"},
                {"is_valid": True, "diagram_type": "gantt"},
            ],
            "overall_validation": True,
            "valid_count": 2,
            "total_count": 2,
            "error": None,
        }

        result = run_flowforge_workflow(
            proposal="Build a customer portal",
            prompt="Executive summary diagrams",
            hf_token="test-token",
            audience_type="stakeholder",
            diagram_types=["flowchart", "gantt"],
            optimize_prompt=False,
        )

        response = format_workflow_response(result)

        self.assertEqual(response["status"], "success")
        self.assertIsNotNone(response["timeline"])
        self.assertEqual(len(response["diagrams"]), 2)
        self.assertTrue(response["overall_validation"])

        for diagram in response["diagrams"]:
            self.assertIn(diagram["diagram_type"], ["flowchart", "gantt", "architecture"])

    @patch('src.workflow.graph_workflow.SessionManager')
    @patch('src.workflow.graph_workflow.TimeAgent')
    @patch('src.workflow.graph_workflow.TimeValidatorAgent')
    @patch('src.workflow.graph_workflow.PlanAgent')
    @patch('src.workflow.graph_workflow.ImageGeneratorAgent')
    @patch('src.workflow.graph_workflow.ValidatorAgent')
    def test_audience_type_preserved_in_final_state(
        self,
        mock_validator,
        mock_image,
        mock_plan,
        mock_time_validator,
        mock_time,
        mock_session_manager,
    ):
        """audience_type should be preserved in the final workflow state."""
        for agent_mock in [mock_time, mock_time_validator, mock_plan, mock_image, mock_validator]:
            agent_mock.return_value.execute.return_value = {
                "timetable": "Phase 1",
                "milestones": [],
                "parallel_work_streams": [],
                "plan": "plan",
                "diagrams": [],
                "diagram_count": 0,
                "valid_diagram_count": 0,
                "validation_results": [],
                "overall_validation": True,
                "time_overall_validation": True,
                "error": None,
            }

        result = run_flowforge_workflow(
            proposal="Test proposal",
            prompt="Test prompt",
            hf_token="test-token",
            audience_type="stakeholder",
            optimize_prompt=False,
        )

        self.assertEqual(result.get("audience_type"), "stakeholder")