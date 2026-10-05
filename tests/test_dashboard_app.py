"""Headless smoke tests for the Streamlit dashboard navigation."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DashboardAppTests(unittest.TestCase):
    def test_support_dashboard_renders_without_breaking_chat_entrypoint(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=10).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Customer Support" in title.value for title in app.title))

        app.radio[0].set_value("Support Dashboard").run()

        self.assertFalse(app.exception)
        self.assertTrue(any("Support Dashboard" in title.value for title in app.title))
        self.assertTrue(any("Topic breakdown" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
