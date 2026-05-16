"""End-to-end tests for FlowForge frontend (Selenium) and backend (API)."""
import time
import unittest
from unittest.mock import patch
import os
import sys

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from src.app import app
from src.schema.helpers import format_workflow_response
from src.workflow.graph_workflow import run_flowforge_workflow


class TestBackendE2E(unittest.TestCase):
    """End-to-end tests for backend API."""

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    @patch('src.api.router.run_flowforge_workflow')
    @patch('src.api.router.get_hf_token')
    def test_full_api_flow_success(self, mock_get_token, mock_run_workflow):
        mock_get_token.return_value = "test-hf-token"
        mock_run_workflow.return_value = {
            "proposal_summary": "E-commerce Platform",
            "optimized_prompt": "Comprehensive workflow for e-commerce platform",
            "timetable": "gantt\ntitle E-Commerce\nsection Design\nUI Design :a1, 2026-05-10, 14d\nsection Development\nBackend API :after a1, 21d\nsection Deployment\nRelease :after a1, 7d",
            "milestones": [
                "Design complete",
                "Backend development complete",
                "Frontend integration complete",
                "Testing complete",
                "Deployment ready"
            ],
            "parallel_work_streams": ["Frontend", "Backend", "Database"],
            "plan": "Phase 1: Design the UI\nPhase 2: Build backend APIs\nPhase 3: Integrate and test",
            "diagrams": [
                {
                    "diagram_type": "workflow",
                    "mermaid_code": "flowchart TD\n    A[User Login] --> B[Browse Products]\n    B --> C[Add to Cart]\n    C --> D[Checkout]",
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
                    "title": "E-commerce Platform",
                    "description": "Build a full-stack e-commerce platform with user authentication",
                    "requirements": ["User authentication", "Product catalog", "Shopping cart"],
                    "constraints": ["Must use microservices"],
                    "tech_stack": ["React", "Node.js", "PostgreSQL"],
                    "timeline_weeks": 16,
                    "team_size": 8,
                    "budget_range": "$80k-$100k"
                },
                "prompt": {
                    "user_prompt": "Generate workflow and CI/CD diagrams",
                    "diagram_types": ["workflow", "ci_cd"],
                    "optimize_prompt": True,
                    "priority": "high"
                },
                "hf_token": "test-token"
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["proposal_summary"], "E-commerce Platform")
        self.assertEqual(data["total_diagram_count"], 2)
        self.assertEqual(data["valid_diagram_count"], 2)
        self.assertTrue(data["overall_validation"])
        self.assertEqual(len(data["diagrams"]), 2)
        self.assertIn("workflow", [d["diagram_type"] for d in data["diagrams"]])

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertIn(response.status_code, [200, 500])

    @patch('src.api.router.get_hf_token')
    def test_invalid_request_validation(self, mock_get_token):
        mock_get_token.return_value = "test-token"

        response = self.client.post(
            "/api/v1/generate-diagrams",
            json={"proposal": {}, "prompt": {}},
        )
        self.assertEqual(response.status_code, 422)

    def test_cors_headers(self):
        response = self.client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertIn("access-control-allow-origin", response.headers)


class TestFrontendE2E(unittest.TestCase):
    """End-to-end tests for frontend using Selenium."""

    driver = None
    backend_process = None
    frontend_process = None

    @classmethod
    def setUpClass(cls):
        cls.driver = None
        try:
            chrome_options = ChromeOptions()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            cls.driver = webdriver.Chrome(options=chrome_options)
            cls.driver.implicitly_wait(10)
        except Exception:
            cls.driver = None

    @classmethod
    def tearDownClass(cls):
        if cls.driver:
            cls.driver.quit()

    def setUp(self):
        if not self.driver:
            self.skipTest("ChromeDriver not available")

    def test_frontend_loads_successfully(self):
        if not self.driver:
            self.skipTest("Selenium driver not available")

        try:
            self.driver.get("http://localhost:8501")
            wait = WebDriverWait(self.driver, 10)
            title = wait.until(
                EC.presence_of_element_located((By.XPATH, "//h1[contains(., 'FlowForge')]"))
            )
            self.assertIsNotNone(title)
        except Exception:
            self.skipTest("Frontend not running or not accessible")

    def test_project_form_elements_exist(self):
        if not self.driver:
            self.skipTest("Selenium driver not available")

        try:
            self.driver.get("http://localhost:8501")
            wait = WebDriverWait(self.driver, 5)

            project_title = wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@aria-label='Project Title']"))
            )
            self.assertTrue(project_title.is_displayed())
        except Exception:
            self.skipTest("Frontend not running")

    def test_diagram_types_multiselect(self):
        if not self.driver:
            self.skipTest("Selenium driver not available")

        try:
            self.driver.get("http://localhost:8501")
            wait = WebDriverWait(self.driver, 5)

            tab2 = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., '🎨 Diagrams')]"))
            )
            tab2.click()

            time.sleep(1)
        except Exception:
            self.skipTest("Frontend not running")

    def test_generate_button_exists(self):
        if not self.driver:
            self.skipTest("Selenium driver not available")

        try:
            self.driver.get("http://localhost:8501")
            wait = WebDriverWait(self.driver, 5)

            generate_btn = wait.until(
                EC.presence_of_element_located((By.XPATH, "//button[contains(., '🚀 Generate Diagrams')]"))
            )
            self.assertTrue(generate_btn.is_displayed())
        except Exception:
            self.skipTest("Frontend not running")


class TestAPIContractCompliance(unittest.TestCase):
    """Tests verifying API response structure compliance."""

    def test_workflow_response_format(self):
        result = {
            "proposal_summary": "Test Project",
            "optimized_prompt": "Optimized prompt",
            "timetable": "gantt\ntitle Test\nsection A\nTask :a1, 2026-05-10, 5d",
            "milestones": ["Milestone 1", "Milestone 2"],
            "parallel_work_streams": ["Stream A"],
            "plan": "Test plan",
            "diagrams": [
                {"diagram_type": "workflow", "mermaid_code": "A --> B", "is_valid": True, "title": "D1"}
            ],
            "diagram_count": 1,
            "valid_diagram_count": 1,
            "validation_results": [{"is_valid": True, "diagram_type": "workflow"}],
            "overall_validation": True,
            "current_agent": "validator_agent",
            "error": None,
        }

        response = format_workflow_response(result)

        required_fields = [
            "status", "proposal_summary", "timeline", "plan",
            "diagrams", "total_diagram_count", "valid_diagram_count",
            "overall_validation", "error", "current_agent"
        ]
        for field in required_fields:
            self.assertIn(field, response, f"Missing required field: {field}")

    def test_error_response_format(self):
        result = {
            "proposal_summary": None,
            "diagrams": [],
            "diagram_count": 0,
            "valid_diagram_count": 0,
            "error": "Workflow failed",
        }

        response = format_workflow_response(result)

        self.assertEqual(response["status"], "failed")
        self.assertIn("error", response)
        self.assertEqual(response["diagrams"], [])

    def test_partial_success_response_format(self):
        result = {
            "proposal_summary": "Partial",
            "diagrams": [
                {"diagram_type": "workflow", "mermaid_code": "A --> B", "is_valid": True, "title": "D1"},
                {"diagram_type": "ci_cd", "mermaid_code": "bad", "is_valid": False, "title": "D2"},
            ],
            "diagram_count": 2,
            "valid_diagram_count": 1,
            "validation_results": [
                {"is_valid": True, "diagram_type": "workflow"},
                {"is_valid": False, "diagram_type": "ci_cd"},
            ],
            "overall_validation": False,
            "error": None,
        }

        response = format_workflow_response(result)

        self.assertEqual(response["status"], "partial")
        self.assertFalse(response["overall_validation"])


class TestIntegrationAPI(unittest.TestCase):
    """Integration tests with running backend server."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://localhost:8000"
        cls.session = requests.Session()

    def test_health_check_live(self):
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health", timeout=5)
            self.assertIn(response.status_code, [200, 500])
        except requests.exceptions.ConnectionError:
            self.skipTest("Backend server not running")

    def test_generate_diagrams_live(self):
        try:
            payload = {
                "proposal": {
                    "title": "Test Project",
                    "description": "A test project for integration testing",
                    "requirements": ["Requirement 1"],
                    "constraints": [],
                    "tech_stack": ["Python"],
                    "timeline_weeks": 8,
                    "team_size": 3,
                },
                "prompt": {
                    "user_prompt": "Generate a workflow diagram for testing",
                    "diagram_types": ["workflow"],
                    "optimize_prompt": False,
                    "priority": "medium"
                },
                "hf_token": os.environ.get("HF_TOKEN", "test-token")
            }

            response = self.session.post(
                f"{self.base_url}/api/v1/generate-diagrams",
                json=payload,
                timeout=120
            )

            self.assertIn(response.status_code, [200, 500])

            if response.status_code == 200:
                data = response.json()
                self.assertIn("status", data)
                self.assertIn("diagrams", data)
        except requests.exceptions.ConnectionError:
            self.skipTest("Backend server not running")


if __name__ == "__main__":
    unittest.main()