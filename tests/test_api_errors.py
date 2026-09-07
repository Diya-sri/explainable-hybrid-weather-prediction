import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ApiErrorSafetyTests(unittest.TestCase):
    def test_unauthorized_error_does_not_expose_key(self):
        from weather_api import validate_status
        with self.assertRaises(RuntimeError) as caught:
            validate_status(401)
        self.assertNotIn("secret-test-key", str(caught.exception))
        self.assertIn("rejected", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
