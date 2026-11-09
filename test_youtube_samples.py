"""Test section commands and decoded pixels without contacting YouTube."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

import cv2
import numpy as np

import collect_youtube_samples as collector
from collect_youtube_samples import youtube_samples


class YouTubeSampleTests(unittest.TestCase):
    def test_newer_launcher_module_is_selected_over_old_path_executable(self):
        calls = []
        def run(command, **kwargs):
            calls.append((command, kwargs))
            version = b'2026.10.26' if '-m' in command else b'2026.03.17'
            return SimpleNamespace(returncode=0, stdout=version, stderr=b'')
        with patch.object(collector, 'IS_WINDOWS', True), patch.object(collector, '_local_candidates', return_value=[['old-yt-dlp.exe']]), patch.object(collector.shutil, 'which', return_value='py.exe'), patch.object(collector.importlib.util, 'find_spec', return_value=None):
            selected = collector.find_tool('yt-dlp', runner=run)
        self.assertEqual(selected, ['py.exe', '-m', 'yt_dlp'])
        self.assertTrue(all(k['shell'] is False and k['timeout']==5 for _, k in calls))

    def test_winget_ffmpeg_is_found_without_refreshed_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            exe = root/'Microsoft/WinGet/Packages/Gyan.FFmpeg_example/ffmpeg-9-full_build/bin/ffmpeg.exe'
            exe.parent.mkdir(parents=True); exe.write_bytes(b'test')
            with patch.object(collector, 'IS_WINDOWS', True), patch.object(collector.shutil, 'which', return_value=None), patch.dict(collector.os.environ, {'LOCALAPPDATA':str(root)}):
                self.assertEqual(collector.find_tool('ffmpeg'), [str(exe)])

    def test_error_details_identify_tool_and_redact_urls_and_authorization(self):
        def run(command, **kwargs):
            self.assertFalse(kwargs['shell'])
            return SimpleNamespace(returncode=1, stdout=b'private metadata should not be logged', stderr=b'ERROR: video not available https://cdn.example/media?signature=SECRET\nAuthorization: Bearer PRIVATE\ncookie=a=COOKIEVALUE; b=VALUE')
        with self.assertRaises(RuntimeError) as result:
            collector._run(['py.exe','-m','yt_dlp','url'],run,10)
        detail = str(result.exception)
        self.assertIn('yt-dlp', detail); self.assertIn('video not available', detail)
        for value in ['SECRET','PRIVATE','COOKIEVALUE','metadata should not','cdn.example']:
            self.assertNotIn(value,detail)

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

    def test_resume_does_not_download_or_decode_completed_sample_indices(self):
        downloads=[];_,png=cv2.imencode('.png',np.zeros((25,40,3),np.uint8))
        def run(command,**kwargs):
            if '--dump-single-json' in command:return SimpleNamespace(returncode=0,stdout=json.dumps({'id':'RzZ0gH6DMF8','duration':100}).encode(),stderr=b'')
            if '--download-sections' in command:
                downloads.append(command[command.index('--download-sections')+1])
                Path(command[command.index('-o')+1].replace('%(ext)s','mp4')).write_bytes(b'fixture')
                return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
            return SimpleNamespace(returncode=0,stdout=png.tobytes(),stderr=b'')
        rows=list(youtube_samples('https://youtu.be/RzZ0gH6DMF8',count=5,skip_indices={1,2,3,4},runner=run,yt_command=['yt-dlp'],ffmpeg_command=['ffmpeg']))
        self.assertEqual(len(downloads),1);self.assertEqual(rows[0][1]['sample_index'],5)


if __name__ == "__main__":
    unittest.main()
