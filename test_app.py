"""Headless integration tests for recording, source resets and video acquisition."""
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np

from app import App, Capture
from measurement import Settings, demo_frame


class FakeValue:
    def __init__(self):
        self.value = ""

    def set(self, value):
        self.value = value

    def get(self):
        return self.value

    def delete(self, *_):
        pass

    def insert(self, *_):
        pass


def headless_app(directory):
    app = App.__new__(App)
    app.capture = None
    app.frame = demo_frame()
    app.source = "test.png"
    app.ppmm = 10
    app.calibration_points = [(700, 450), (800, 450)]
    app.roi = None
    app.position = None
    app.mode = "auto"
    app.points = []
    app.settings = Settings()
    app.paused = app.ended = False
    app.data = Path(directory)
    app.csv_path = app.data / "measurements.csv"
    app.obs_path = app.data / "obs_status.txt"
    app.io_error = ""
    app.items = FakeValue()
    app.status = FakeValue()
    app.result = FakeValue()
    app.render = lambda: None
    app.show_mask = FakeValue()
    app.ai_worker = None
    app.ai_mode = "off"
    app.ai_pending = False
    app.new_series()
    return app


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class ApplicationTests(unittest.TestCase):
    def test_validation_sample_preserves_original_crop_pixels_and_scale_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            app.roi = (50, 30, 650, 400)
            path = app.save_sample()
            self.assertIsNotNone(path)
            self.assertEqual(path.suffix, ".zip")
            with ZipFile(path) as archive:
                self.assertEqual(len(archive.namelist()), 2)
                saved = cv2.imdecode(np.frombuffer(archive.read(path.stem+".png"), dtype=np.uint8), cv2.IMREAD_COLOR)
                metadata = json.loads(archive.read(path.stem+".json"))
            np.testing.assert_array_equal(saved, app.frame[30:400, 50:650])
            self.assertEqual(list((app.data / "samples").iterdir()), [path])
            self.assertEqual(metadata["crop_xyxy"], [50, 30, 650, 400])
            self.assertEqual(metadata["source_resolution"], [960, 540])
            self.assertEqual(metadata["pixels_per_mm"], 10)
            self.assertEqual(metadata["settings"]["appearance"], "light")
            self.assertFalse(app.csv_path.exists())

    def test_failed_sample_export_leaves_no_incomplete_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            with patch("app.os.replace", side_effect=OSError("test write failure")):
                self.assertIsNone(app.save_sample())
            self.assertEqual(list((app.data / "samples").iterdir()), [])
            self.assertIn("保存に失敗", app.status.value)

    def test_auto_csv_scale_and_obs_output(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            app.process()
            rows = read_rows(app.csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["unit"], "mm")
            self.assertEqual(rows[0]["method"], "auto_candidate")
            self.assertEqual(rows[0]["count"], "2")
            obs = app.obs_path.read_text(encoding="utf-8-sig")
            self.assertIn("蚕の認識は未確認", obs)
            self.assertIn("体長の測定値なし", obs)
            self.assertNotIn("平均体長", obs)
            self.assertNotIn("匹", obs)
            self.assertIn("未確認候補", app.result.value)
            app.process()  # The same frame must not produce a second automatic row immediately.
            self.assertEqual(len(read_rows(app.csv_path)), 1)

    def test_source_change_invalidates_scale_roi_history_and_old_obs(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            app.process()
            old_series = app.series
            app.roi = (10, 10, 900, 500)
            app.reset_source("camera:1")
            self.assertIsNone(app.ppmm)
            self.assertIsNone(app.roi)
            self.assertIsNone(app.frame)
            self.assertEqual(app.calibration_points, [])
            self.assertEqual(app.history, [])
            self.assertNotEqual(old_series, app.series)
            self.assertIn("測定値なし", app.obs_path.read_text(encoding="utf-8-sig"))

    def test_paused_frame_is_not_recorded_as_new_automatic_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            app.paused = True
            app.process()
            self.assertFalse(app.csv_path.exists())
            self.assertIn("停止・静止値", app.obs_path.read_text(encoding="utf-8-sig"))

    def test_manual_polyline_is_recorded_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            app.mode = "manual"
            app.points = [(0, 0), (30, 40), (60, 0)]
            app.finish_manual()
            rows = read_rows(app.csv_path)
            self.assertEqual(rows[0]["method"], "manual")
            self.assertEqual(float(rows[0]["mean"]), 10)
            self.assertEqual(app.history, [])

    def test_video_reader_delivers_frames_then_reports_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "test.avi")
            frame = demo_frame()
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (960, 540))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(frame)
            writer.release()
            reader = Capture(path)
            try:
                sample, position, acquired = reader.frames.get(timeout=3)
                self.assertEqual(sample.shape, frame.shape)
                self.assertIsNotNone(position)
                deadline = time.monotonic() + 3
                while reader.terminal is None and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertIn("終端", reader.terminal or "")
            finally:
                reader.close()
            self.assertFalse(reader.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
