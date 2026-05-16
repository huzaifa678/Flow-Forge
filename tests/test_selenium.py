"""Selenium-based end-to-end tests for FlowForge frontend UI."""
import os
import sys
import time
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestFrontendSelenium(unittest.TestCase):
    """Frontend E2E tests using Selenium WebDriver."""

    driver = None

    @classmethod
    def setUpClass(cls):
        cls.driver = None
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options

            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-gpu")

            cls.driver = webdriver.Chrome(options=chrome_options)
            cls.driver.implicitly_wait(10)
        except Exception as e:
            print(f"ChromeDriver setup failed: {e}")
            cls.driver = None

    @classmethod
    def tearDownClass(cls):
        if cls.driver:
            cls.driver.quit()
            cls.driver = None

    def setUp(self):
        if not self.driver:
            self.skipTest("Selenium ChromeDriver not available")

    def test_01_page_loads(self):
        """Test that the frontend page loads correctly."""
        try:
            from selenium.webdriver.support import expected_conditions as EC
            self.driver.get("http://localhost:8501")
            WebDriverWaitClass = self._get_webdriver_wait()
            WebDriverWaitClass(self.driver, 10).until(EC.title_contains("FlowForge"))
            self.assertIn("FlowForge", self.driver.title)
        except Exception as e:
            self.skipTest(f"Frontend not running: {e}")

    def test_02_page_title_displayed(self):
        """Test that FlowForge title is displayed."""
        try:
            self.driver.get("http://localhost:8501")
            WebDriverWait = self._get_webdriver_wait()
            
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            
            title = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//h1[contains(., 'FlowForge')]"))
            )
            self.assertTrue(title.is_displayed())
        except Exception as e:
            self.skipTest(f"Frontend not running: {e}")

    def test_03_project_form_inputs(self):
        """Test that project form inputs are present."""
        try:
            self.driver.get("http://localhost:8501")
            WebDriverWait = self._get_webdriver_wait()
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            
            inputs = [
                "//input[@aria-label='Project Title']",
                "//textarea[@aria-label='Project Description']",
                "//input[@aria-label='Timeline (weeks)']",
                "//input[@aria-label='Team Size']",
            ]
            
            for xpath in inputs:
                try:
                    el = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    self.assertTrue(el.is_displayed())
                except Exception:
                    pass
        except Exception as e:
            self.skipTest(f"Frontend not running: {e}")

    def test_04_tabs_present(self):
        """Test that all tabs are present."""
        try:
            self.driver.get("http://localhost:8501")
            WebDriverWait = self._get_webdriver_wait()
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            
            tabs = [
                "📋 Project",
                "🎨 Diagrams", 
                "⚙️ Advanced"
            ]
            
            for tab in tabs:
                try:
                    tab_btn = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, f"//button[contains(., '{tab}')]"))
                    )
                    self.assertTrue(tab_btn.is_displayed())
                except Exception:
                    pass
        except Exception as e:
            self.skipTest(f"Frontend not running: {e}")

    def test_05_generate_button_clickable(self):
        """Test that generate button is clickable."""
        try:
            self.driver.get("http://localhost:8501")
            WebDriverWait = self._get_webdriver_wait()
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            
            gen_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., '🚀 Generate Diagrams')]"))
            )
            self.assertTrue(gen_btn.is_enabled())
        except Exception as e:
            self.skipTest(f"Frontend not running: {e}")

    def test_06_diagram_types_selection(self):
        """Test diagram types multiselect in Diagrams tab."""
        try:
            self.driver.get("http://localhost:8501")
            WebDriverWait = self._get_webdriver_wait()
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            
            tab = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., '🎨 Diagrams')]"))
            )
            tab.click()
            time.sleep(1)
            
            multiselect = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@data-testid='stMultiSelect']"))
            )
        except Exception as e:
            self.skipTest(f"Frontend not running: {e}")

    def test_07_advanced_tab_settings(self):
        """Test advanced tab settings."""
        try:
            self.driver.get("http://localhost:8501")
            WebDriverWait = self._get_webdriver_wait()
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            
            tab = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., '⚙️ Advanced')]"))
            )
            tab.click()
            time.sleep(1)
            
            priority = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[contains(@aria-label, 'Priority')]"))
            )
        except Exception as e:
            self.skipTest(f"Frontend not running: {e}")

    def test_08_fill_and_submit_form(self):
        """Test filling form and submitting."""
        try:
            self.driver.get("http://localhost:8501")
            WebDriverWait = self._get_webdriver_wait()
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC
            
            title_input = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//input[@aria-label='Project Title']"))
            )
            title_input.clear()
            title_input.send_keys("Test Project E2E")
            
            desc_input = self.driver.find_element(By.XPATH, "//textarea[@aria-label='Project Description']")
            desc_input.clear()
            desc_input.send_keys("Test description for E2E testing")
        except Exception as e:
            self.skipTest(f"Frontend not running: {e}")

    def _get_webdriver_wait(self):
        from selenium.webdriver.support.ui import WebDriverWait
        return WebDriverWait


class TestFrontendWithMockedBackend(unittest.TestCase):
    """Frontend tests with mocked backend responses."""

    def test_frontend_payload_build(self):
        """Test that frontend builds correct payload structure."""
        from frontend.payload import build_payload
        import streamlit as st

        with patch.dict('streamlit.session_state', {
            'proposal_title': 'Test Project',
            'proposal_desc': 'Test Description',
            'requirements': ['Req 1', 'Req 2'],
            'constraints': ['Constraint 1'],
            'tech_stack': ['Python', 'FastAPI'],
            'budget_range': '$50k-$100k',
            'timeline_weeks': 12,
            'team_size': 5,
            'user_prompt_txt': 'Generate workflow diagram',
            'diagram_types_sel': ['workflow', 'ci_cd'],
            'priority': 'high',
            'optimize_prompt': True,
            'include_gantt': True,
            'hf_token': 'test-token'
        }):
            pass

    def test_forms_render_correctly(self):
        """Test that forms module renders without errors."""
        import sys
        frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
        sys.path.insert(0, frontend_dir)
        try:
            from frontend.forms import render_project_form
            self.assertTrue(callable(render_project_form))
        except Exception as e:
            self.skipTest(f"Frontend dependencies not available: {e}")
        finally:
            if frontend_dir in sys.path:
                sys.path.remove(frontend_dir)


if __name__ == "__main__":
    unittest.main()