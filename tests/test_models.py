import unittest
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explainability import exact_shapley
from hybrid_model import NumpyLSTM, hybrid_probabilities


class HybridModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(ROOT / "hybrid_bundle.joblib")
        cls.lstm = NumpyLSTM.load(ROOT / "lstm_model.npz")
        b = cls.bundle["baselines"]
        cls.sequence = np.broadcast_to(
            b, (1, cls.bundle["sequence_length"], len(b))
        ).copy()
        cls.sequence[0, -1, 2:] += 0.25

    def test_probabilities_are_normalized(self):
        p = hybrid_probabilities(
            self.bundle["tree"], self.lstm, self.sequence,
            self.bundle["blend_weight"]
        )
        self.assertEqual(p.shape, (1, 8))
        self.assertTrue(np.all(p >= 0))
        self.assertAlmostEqual(float(p.sum()), 1.0, places=12)

    def test_lstm_is_deterministic_after_serialization(self):
        first = self.lstm.predict_proba(self.sequence)
        second = NumpyLSTM.load(ROOT / "lstm_model.npz").predict_proba(self.sequence)
        np.testing.assert_allclose(first, second, rtol=0, atol=0)

    def test_exact_shapley_additivity(self):
        p = hybrid_probabilities(
            self.bundle["tree"], self.lstm, self.sequence,
            self.bundle["blend_weight"]
        )[0]
        selected = int(p.argmax())
        values, baseline, prediction = exact_shapley(
            self.bundle["tree"], self.lstm, self.sequence,
            self.bundle["blend_weight"], self.bundle["baselines"], selected
        )
        self.assertEqual(len(values), 11)
        self.assertAlmostEqual(baseline + float(values.sum()), prediction, places=12)
        self.assertAlmostEqual(prediction, float(p[selected]), places=12)


if __name__ == "__main__":
    unittest.main()
