"""Test geometry, offsets and worker behavior without downloading or importing Torch."""
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

from ai_measurement import AIJob, AIWorker, LocalSegmenter, candidates_from_masks, content_view
from app import App
from measurement import Settings, analyze, summarize
from test_app import FakeValue, headless_app, read_rows


def capsule():
    frame = np.full((300, 500, 3), 255, np.uint8)
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.line(mask, (100, 120), (300, 120), 1, 20)
    frame[mask > 0] = 230
    region = {"segmentation": mask, "predicted_iou": .97, "stability_score": .98}
    return frame, region


class MaskGeometryTests(unittest.TestCase):
    def test_independent_body_mask_does_not_merge_with_white_background(self):
        frame, region = capsule()
        baseline, _ = analyze(frame)
        self.assertEqual(summarize(baseline)["count"], 0)
        candidates = candidates_from_masks(frame, [region], Settings())
        self.assertEqual(summarize(candidates)["count"], 1)
        self.assertAlmostEqual(candidates[0].length_px, 220, delta=6)

    def test_roi_coordinates_are_restored_to_source_frame(self):
        frame, region = capsule()
        roi = (50, 40, 450, 250)
        view, offset = content_view(frame, roi)
        region["segmentation"] = region["segmentation"][40:250, 50:450]
        candidates = candidates_from_masks(view, [region], Settings(), offset)
        self.assertEqual(summarize(candidates)["count"], 1)
        self.assertAlmostEqual(float(np.mean(candidates[0].path[:, 1])), 120, delta=1)
        self.assertGreater(candidates[0].path[:, 0].max(), 305)

    def test_letterbox_cut_body_is_excluded(self):
        frame, region = capsule()
        frame[160:] = 0
        cv2.line(frame, (390, 80), (390, 159), (230, 230, 230), 18)
        frame[160:] = 0
        view, offset = content_view(frame)
        self.assertEqual(view.shape[0], 160)
        mask = np.zeros(view.shape[:2], np.uint8)
        cv2.line(mask, (390, 80), (390, 159), 1, 18)
        cut = {"segmentation": mask, "predicted_iou": .97, "stability_score": .98}
        candidates = candidates_from_masks(view, [cut], Settings(), offset)
        self.assertEqual(summarize(candidates)["count"], 0)
        self.assertEqual(candidates[0].reason, "edge")

    def test_nested_masks_do_not_double_count_the_same_body(self):
        frame, region = capsule()
        small = region["segmentation"].copy()
        small[:, 230:] = 0
        part = {**region, "segmentation": small}
        candidates = candidates_from_masks(frame, [part, region], Settings())
        self.assertEqual(summarize(candidates)["count"], 1)
        self.assertIn("duplicate_or_part", [c.reason for c in candidates])

    def test_all_black_input_has_no_content(self):
        view, _ = content_view(np.zeros((100, 100, 3), np.uint8))
        self.assertIsNone(view)

    def test_missing_checkpoint_reports_the_setup_action(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "setup_ai_windows.bat"):
                LocalSegmenter(Path(directory) / "missing.pt")


class AsyncTests(unittest.TestCase):
    def test_inference_does_not_block_submission_and_cannot_accumulate_jobs(self):
        release = threading.Event()

        class SlowEngine:
            def analyze(self, frame, settings, roi):
                release.wait(timeout=3)
                return [], "fake-cpu"

        worker = AIWorker(SlowEngine)
        frame, _ = capsule()
        job = AIJob(frame, None, Settings(), None, "old-context")
        try:
            start = time.monotonic()
            self.assertTrue(worker.submit(job))
            self.assertLess(time.monotonic()-start, .5)
            self.assertFalse(worker.submit(job))
            release.set()
            result = worker.results.get(timeout=3)
            self.assertEqual(result["job"].token, "old-context")
            self.assertEqual(result["device"], "fake-cpu")
        finally:
            release.set()
            worker.close()
            worker.thread.join(timeout=1)

    def test_stale_ai_result_cannot_replace_a_new_source(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            old_frame = app.frame.copy()
            old_token = (app.series, app.ai_generation)
            app.new_series()
            app.ai_mode, app.ai_pending = "single", False
            job = AIJob(old_frame, None, Settings(), None, old_token)

            class Finished:
                def receive(self):
                    return {"job": job, "error": None, "candidates": [], "device": "cpu", "seconds": 1}

            app.ai_worker = Finished()
            current = app.frame
            app.poll_ai()
            self.assertIs(app.frame, current)
            self.assertFalse(app.csv_path.exists())

    def test_result_is_drawn_on_its_own_frame_and_recorded_as_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            frame, region = capsule()
            candidates = candidates_from_masks(frame, [region], Settings())
            job = AIJob(frame, 12.0, Settings(), None, (app.series, app.ai_generation))

            class Finished:
                def receive(self):
                    return {"job": job, "error": None, "candidates": candidates, "device": "cpu", "seconds": 1}

            app.ai_mode, app.ai_pending, app.paused = "single", False, True
            app.ai_worker, app.show_mask = Finished(), FakeValue()
            app.poll_ai()
            self.assertIs(app.frame, frame)
            self.assertEqual(app.position, 12)
            self.assertIn("AI静止結果", app.result.value)
            self.assertEqual(read_rows(app.csv_path)[0]["method"], "sam_auto_candidate")
            self.assertIn("体長の測定値なし", app.obs_path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
