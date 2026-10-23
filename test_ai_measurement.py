"""Test geometry, offsets and worker behavior without downloading or importing Torch."""
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

from ai_measurement import AIJob, AIWorker, LocalSegmenter, candidates_from_masks, content_view, prompted_candidates
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


class PromptTests(unittest.TestCase):
    def test_all_positive_and_negative_constraints_are_required(self):
        frame, region = capsule()
        mask = region["segmentation"].astype(bool)
        wrong = mask.copy()
        wrong[80:140, 350:410] = True
        candidates = prompted_candidates(frame, [wrong, mask], [.99, .8],
                                          [(150, 120), (380, 100)], [1, 0], Settings())
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].length_px, 220, delta=6)

    def test_prompt_roi_offsets_and_out_of_range_points(self):
        frame, region = capsule()
        view = frame[40:250, 50:450]
        mask = region["segmentation"][40:250, 50:450]
        candidates = prompted_candidates(view, [mask], [.9], [(150, 120)], [1], Settings(), (50, 40))
        self.assertAlmostEqual(float(candidates[0].path[:, 1].mean()), 120, delta=1)
        with self.assertRaisesRegex(ValueError, "測定範囲"):
            prompted_candidates(view, [mask], [.9], [(30, 120)], [1], Settings(), (50, 40))
        with self.assertRaisesRegex(ValueError, "少なくとも1点"):
            prompted_candidates(view, [mask], [.9], [(150, 120)], [0], Settings(), (50, 40))

    def test_disconnected_positive_points_are_never_joined(self):
        frame, region = capsule()
        mask = region["segmentation"].copy()
        cv2.line(mask, (380, 100), (380, 200), 1, 18)
        candidates = prompted_candidates(frame, [mask], [.9],
                                          [(150, 120), (380, 150)], [1, 1], Settings())
        self.assertEqual(candidates, [])

    def test_unrelated_island_is_removed_but_long_crossing_is_still_rejected(self):
        frame, region = capsule()
        mask = region["segmentation"].copy()
        mask[60:65, 400:405] = 1
        candidates = prompted_candidates(frame, [mask], [.9], [(150, 120)], [1], Settings())
        self.assertIsNotNone(candidates[0].length_px)
        cv2.line(mask, (200, 60), (200, 200), 1, 18)
        candidates = prompted_candidates(frame, [mask], [.9], [(150, 120)], [1], Settings())
        self.assertIsNone(candidates[0].length_px)

    def test_embedding_cache_is_reused_for_points_but_not_changed_image_pixels(self):
        frame, region = capsule()

        class Predictor:
            calls = 0

            def set_image(self, image):
                self.calls += 1

            def predict(self, **kwargs):
                return [region["segmentation"]], [.9], None

        engine = LocalSegmenter.__new__(LocalSegmenter)
        engine.predictor = Predictor()
        engine.prompt_image_key = None
        engine.device = "fake"
        engine.prompt(frame, Settings(), None, [(150, 120)], [1])
        engine.prompt(frame.copy(), Settings(), None, [(170, 120)], [1])
        self.assertEqual(engine.predictor.calls, 1)
        changed = frame.copy()
        changed[0, 0] = 0
        engine.prompt(changed, Settings(), None, [(170, 120)], [1])
        self.assertEqual(engine.predictor.calls, 2)

    def test_worker_routes_prompt_jobs_without_running_automatic_generator(self):
        class PromptEngine:
            def prompt(self, frame, settings, roi, points, labels):
                return [(points, labels)], "fake"

        worker = AIWorker(PromptEngine)
        frame, _ = capsule()
        job = AIJob(frame, None, Settings(), None, "prompt", ((150, 120),), (1,))
        try:
            self.assertTrue(worker.submit(job))
            result = worker.results.get(timeout=3)
            self.assertIsNone(result["error"])
            self.assertEqual(result["candidates"], [(((150, 120),), (1,))])
        finally:
            worker.close()
            worker.thread.join(timeout=1)

    def prepared_app(self, directory):
        app = headless_app(directory)
        app.begin("ai_point")
        app.frame, region = capsule()
        app.points, app.prompt_labels = [(150, 120)], [1]
        app.prompt_candidates = prompted_candidates(app.frame, [region["segmentation"]], [.9],
                                                     app.points, app.prompt_labels, Settings())
        app.show_prompt(0)
        return app

    def test_prompt_is_not_logged_until_user_confirms_and_never_double_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.prepared_app(directory)
            self.assertFalse(app.csv_path.exists())
            self.assertIn("体長の測定値なし", app.obs_path.read_text(encoding="utf-8-sig"))
            app.finish_prompt()
            app.finish_prompt()
            rows = read_rows(app.csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["method"], "sam_assisted")
            self.assertEqual(rows[0]["count"], "1")
            self.assertEqual(app.history, [])

    def test_saved_sample_retains_prompts_outline_and_user_confirmation(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            app = self.prepared_app(directory)
            app.finish_prompt()
            path = app.save_sample()
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            selection = metadata["assisted_selection"]
            self.assertEqual(selection["status"], "user_confirmed")
            self.assertEqual(selection["prompt_labels"], [1])
            self.assertEqual(selection["prompt_points_source_xy"], [[150, 120]])
            self.assertGreater(len(selection["centerline_source_xy"]), 2)

    def test_adding_a_point_invalidates_inflight_result_and_previous_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.prepared_app(directory)
            job = AIJob(app.frame.copy(), None, Settings(), None,
                        (app.series, app.ai_generation), tuple(app.points), tuple(app.prompt_labels))
            previous = app.prompt_candidates
            app.invalidate_prompt()
            app.poll_prompt({"job": job, "error": None, "candidates": previous})
            self.assertEqual(app.prompt_candidates, [])
            self.assertIsNone(app.prompt_selected)
            self.assertFalse(app.prompt_saved)

    def test_new_worm_clears_previous_points_selection_and_unsavable_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            app = self.prepared_app(directory)
            app.prompt_candidates[0].path = None
            app.prompt_candidates[0].length_px = None
            app.finish_prompt()
            self.assertFalse(app.csv_path.exists())
            app.begin("ai_point")
            self.assertEqual(app.points, [])
            self.assertEqual(app.prompt_labels, [])
            self.assertIsNone(app.prompt_selected)


if __name__ == "__main__":
    unittest.main()
