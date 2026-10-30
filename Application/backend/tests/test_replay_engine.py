from __future__ import annotations

import unittest

import numpy as np

from app.config import DEMO_CLIPS
from app.engines import ReplayEngine


class ReplayEngineTests(unittest.TestCase):
    def test_demo_assets_exist(self) -> None:
        self.assertEqual(len(DEMO_CLIPS), 18)
        self.assertTrue(all(clip.path.is_file() for clip in DEMO_CLIPS))

    def test_recorded_prediction_contract(self) -> None:
        engine = ReplayEngine()
        clip = DEMO_CLIPS[0]
        engine.reset(clip.filename)
        result = engine.predict(np.zeros((8, 8, 3), dtype=np.uint8), 0, clip.filename)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)
        self.assertIn("people", result)
        self.assertIn("algorithm_ms", result)


if __name__ == "__main__":
    unittest.main()
