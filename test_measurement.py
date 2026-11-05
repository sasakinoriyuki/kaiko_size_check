"""Synthetic geometry checks. These do not validate real-silkworm detection accuracy."""
import math
import unittest

import cv2
import numpy as np

from measurement import Settings, analyze, calibrate, channel_roi, demo_frame, polyline_length, summarize


def background():
    return np.full((420, 520, 3), (35, 65, 35), np.uint8)


class GeometryTests(unittest.TestCase):
    def test_calibration_and_manual_curved_polyline(self):
        ppmm = calibrate([(10, 10), (110, 10)], 10)
        self.assertEqual(ppmm, 10)
        self.assertEqual(polyline_length([(0, 0), (30, 40), (60, 0)]) / ppmm, 10)
        for known in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calibrate([(0, 0), (100, 0)], known)

    def test_straight_capsule_full_length(self):
        image = background()
        cv2.line(image, (100, 120), (300, 120), (235, 235, 235), 20)
        candidates, _ = analyze(image)
        stats = summarize(candidates, pixels_per_mm=10)
        self.assertEqual(stats["count"], 1)
        self.assertAlmostEqual(stats["mean"], 22, delta=.6)

    def test_bent_body_length_follows_curve(self):
        image = background()
        # A semicircular tube: full centerline length = pi*100 + two 8px end caps.
        points = np.array([(260+100*math.cos(t), 220+100*math.sin(t))
                           for t in np.linspace(math.pi, 0, 250)], np.int32)
        cv2.polylines(image, [points], False, (235, 235, 235), 16)
        candidates, _ = analyze(image)
        stats = summarize(candidates)
        self.assertEqual(stats["count"], 1, [(c.reason, c.length_px) for c in candidates])
        self.assertAlmostEqual(stats["mean"], math.pi*100+16, delta=13)
        self.assertGreater(stats["mean"], 300)  # A straight bounding-box length would be ~216.

    def test_crossing_is_excluded_not_counted_as_one(self):
        image = background()
        cv2.line(image, (100, 100), (300, 300), (235, 235, 235), 20)
        cv2.line(image, (100, 300), (300, 100), (235, 235, 235), 20)
        candidates, _ = analyze(image)
        self.assertEqual(summarize(candidates)["count"], 0)
        self.assertEqual(candidates[0].reason, "branch_or_loop")

    def test_cut_body_at_frame_or_roi_edge_is_excluded(self):
        image = background()
        cv2.line(image, (0, 100), (200, 100), (235, 235, 235), 20)
        candidates, _ = analyze(image)
        self.assertEqual(candidates[0].reason, "edge")
        image = background()
        cv2.line(image, (80, 100), (300, 100), (235, 235, 235), 20)
        candidates, _ = analyze(image, roi=(150, 50, 400, 200))
        self.assertEqual(candidates[0].reason, "edge")

    def test_no_detections_is_missing_value_not_zero_length(self):
        candidates, _ = analyze(background())
        stats = summarize(candidates)
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["mean"])

    def test_roi_keeps_original_image_coordinates(self):
        image = background()
        cv2.line(image, (100, 100), (300, 100), (235, 235, 235), 20)
        full, _ = analyze(image)
        cropped, _ = analyze(image, roi=(50, 50, 400, 200))
        self.assertAlmostEqual(full[0].length_px, cropped[0].length_px, places=5)
        np.testing.assert_allclose(full[0].path, cropped[0].path)

    def test_synthetic_demo_has_two_valid_bodies_and_one_overlap_region(self):
        candidates, _ = analyze(demo_frame())
        self.assertEqual(summarize(candidates)["count"], 2)
        self.assertEqual(summarize(candidates)["excluded_regions"], 1)

    def test_obs_preset_excludes_duplicate_in_secondary_camera_at_two_resolutions(self):
        for width, height in ((898, 511), (1333, 755)):
            with self.subTest(width=width):
                frame = np.full((height, width, 3), (35, 65, 35), np.uint8)
                cv2.line(frame, (100, 120), (300, 120), (235, 235, 235), 20)
                cv2.line(frame, (100, 240), (300, 240), (235, 235, 235), 20)
                # A copy in the secondary view must not enter the main-view mean/count.
                cv2.line(frame, (int(width*.82), height-80),
                         (int(width*.95), height-80), (235, 235, 235), 14)
                full, _ = analyze(frame)
                main, _ = analyze(frame, roi=channel_roi(frame.shape))
                self.assertEqual(summarize(full)["count"], 3)
                self.assertEqual(summarize(main)["count"], 2)

    def test_dark_candidate_mode_excludes_white_body_and_round_dark_region(self):
        frame = np.full((420, 520, 3), (35, 120, 35), np.uint8)
        cv2.line(frame, (100, 100), (300, 100), (30, 30, 30), 12)
        cv2.line(frame, (100, 220), (300, 220), (235, 235, 235), 12)
        cv2.circle(frame, (380, 300), 18, (30, 30, 30), -1)
        dark, _ = analyze(frame, Settings(appearance="dark", brightness=60))
        light, _ = analyze(frame)
        self.assertEqual(summarize(dark)["count"], 1)
        self.assertEqual(summarize(light)["count"], 1)
        self.assertLess(np.mean([c.path[:, 1].mean() for c in dark if c.path is not None]), 150)
        self.assertGreater(np.mean([c.path[:, 1].mean() for c in light if c.path is not None]), 150)

    def test_two_thirds_layout_excludes_panel_inside_older_wide_preset(self):
        frame = np.full((720, 1080, 3), (35, 120, 35), np.uint8)
        cv2.line(frame, (100, 120), (300, 120), (235, 235, 235), 16)
        # An elongated bright UI element lies between 2/3 and 77% of the width.
        cv2.line(frame, (760, 120), (760, 360), (235, 235, 235), 16)
        wide, _ = analyze(frame, roi=channel_roi(frame.shape))
        narrow, _ = analyze(frame, roi=channel_roi(frame.shape, "two_thirds"))
        self.assertEqual(summarize(wide)["count"], 2)
        self.assertEqual(summarize(narrow)["count"], 1)


if __name__ == "__main__":
    unittest.main()
