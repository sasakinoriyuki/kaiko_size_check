"""Test section commands and decoded pixels without contacting YouTube."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from collect_youtube_samples import youtube_samples


class YouTubeSampleTests(unittest.TestCase):
    def test_downloads_only_sections_and_preserves_decoded_image_size(self):
        commands, temporary_files = [], []
        original = np.full((72, 128, 3), (31, 80, 120), np.uint8)
        _, encoded = cv2.imencode(".png", original)

        def run(command, **kwargs):
            commands.append(command)
            if "--dump-single-json" in command:
                data = {"id": "RzZ0gH6DMF8", "title": "蚕1齢", "duration": 100}
                return SimpleNamespace(returncode=0, stdout=json.dumps(data).encode(), stderr=b"")
            if "--download-sections" in command:
                template = command[command.index("-o")+1]
                path = Path(template.replace("%(ext)s", "mp4"))
                path.write_bytes(b"test video")
                temporary_files.append(path)
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
            self.assertIn("image2pipe", command)
            self.assertNotIn("-vf", command)
            return SimpleNamespace(returncode=0, stdout=encoded.tobytes(), stderr=b"")

        result = list(youtube_samples("https://www.youtube.com/watch?v=RzZ0gH6DMF8", count=5,
                                     runner=run, yt_command=["yt-dlp"], ffmpeg_command=["ffmpeg"]))
        self.assertEqual(len(result), 5)
        downloads = [c for c in commands if "--download-sections" in c]
        self.assertEqual(len(downloads), 5)
        for image, metadata in result:
            np.testing.assert_array_equal(image, original)
            self.assertEqual(metadata["source_kind"], "youtube_archive")
            self.assertAlmostEqual(metadata["requested_section_end"]-metadata["requested_section_start"], 3)
            self.assertIn("Approximate", metadata["timing_note"])
        self.assertEqual([m["requested_video_seconds"] for _, m in result], [10, 30, 50, 70, 90])
        self.assertTrue(all(not path.exists() for path in temporary_files))

    def test_rejects_live_streams_and_playlists_before_any_download(self):
        for metadata in [{"duration": 100, "is_live": True}, {"duration": 100, "_type": "playlist"}, {"duration": None}]:
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0, stdout=json.dumps(metadata).encode(), stderr=b"")

            with self.assertRaisesRegex(ValueError, "完了済み"):
                list(youtube_samples("https://youtu.be/RzZ0gH6DMF8", runner=run,
                                     yt_command=["yt-dlp"], ffmpeg_command=["ffmpeg"]))
            self.assertEqual(len(calls), 1)

    def test_only_accepts_youtube_video_urls(self):
        for url in ["file:///tmp/video.mp4", "https://youtube.com.example.org/video", "https://example.org"]:
            with self.assertRaisesRegex(ValueError, "YouTube"):
                list(youtube_samples(url, yt_command=["yt-dlp"], ffmpeg_command=["ffmpeg"]))


if __name__ == "__main__":
    unittest.main()
