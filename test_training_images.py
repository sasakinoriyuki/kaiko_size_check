"""Verify local sampling, original pixels, split ZIPs and failure cleanup."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import cv2
import numpy as np

from prepare_training_images import image_samples, prepare_archives, sample_indices, video_samples


class TrainingImageTests(unittest.TestCase):
    def test_short_recordings_have_distinct_in_range_frame_positions(self):
        self.assertEqual(sample_indices(1, 5), [0])
        self.assertEqual(sample_indices(2, 5), [0, 1])
        self.assertEqual(sample_indices(101, 5), [10, 30, 50, 70, 90])
        self.assertEqual(sample_indices(101, 1), [50])
        with self.assertRaises(ValueError):
            sample_indices(0)

    def test_zip_contains_original_pixels_and_unlabelled_metadata_without_full_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original 日本語.png"
            image = np.random.default_rng(42).integers(0, 256, (80, 120, 3), dtype=np.uint8)
            ok, data = cv2.imencode(".png", image)
            self.assertTrue(ok)
            source.write_bytes(data.tobytes())
            archives = prepare_archives(image_samples([source]), 2, root / "output")
            self.assertEqual(len(archives), 1)
            with ZipFile(archives[0]) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["declared_instar"], 2)
                self.assertEqual(manifest["annotation_status"], "unlabelled")
                record = manifest["frames"][0]
                self.assertEqual(record["source_file"], source.name)
                self.assertEqual(record["source_resolution"], [120, 80])
                decoded = cv2.imdecode(np.frombuffer(archive.read(record["file"]), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(decoded, image)
            self.assertEqual(source.read_bytes(), data.tobytes())
            self.assertNotIn(str(root), json.dumps(manifest))

    def test_large_batches_split_without_losing_or_duplicate_images(self):
        rng = np.random.default_rng(42)
        images = [rng.integers(0, 256, (50, 60, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            samples = ((image, {"source_file": f"image{i}.png"}) for i, image in enumerate(images))
            paths = prepare_archives(samples, 1, directory, max_payload=10000)
            self.assertEqual(len(paths), 3)
            files = []
            for i, path in enumerate(paths, 1):
                with ZipFile(path) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                    self.assertEqual(manifest["part"], i)
                    self.assertEqual(manifest["parts"], 3)
                    files.extend(entry["file"] for entry in manifest["frames"])
            self.assertEqual(len(set(files)), 3)

    def test_failed_write_rolls_back_new_parts_and_preserves_existing_files(self):
        image = np.zeros((30, 40, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.txt"
            existing.write_text("keep")
            with patch("prepare_training_images.os.replace", side_effect=OSError("test error")):
                with self.assertRaises(OSError):
                    prepare_archives([(image, {"source_file": "sample.png"})], 3, directory)
            self.assertEqual(list(Path(directory).iterdir()), [existing])

    def test_cancelled_preparation_does_not_create_archives(self):
        stop = threading.Event()
        stop.set()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "中止"):
                prepare_archives([(np.zeros((20, 20, 3), np.uint8), {"source_file": "x.png"})], 1, directory, cancelled=stop)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_video_sampling_records_decoded_frames_and_original_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
            self.assertTrue(writer.isOpened())
            for i in range(21):
                writer.write(np.full((120, 160, 3), i*10, np.uint8))
            writer.release()
            samples = list(video_samples(path))
            self.assertEqual(len(samples), 5)
            self.assertEqual([s[1]["decoded_frame"] for s in samples], [2, 6, 10, 14, 18])
            for image, metadata in samples:
                self.assertEqual(image.shape, (120, 160, 3))
                self.assertAlmostEqual(float(image.mean()), metadata["decoded_frame"]*10, delta=2)
                self.assertAlmostEqual(metadata["video_seconds"], metadata["decoded_frame"]/10, delta=.01)


if __name__ == "__main__":
    unittest.main()
