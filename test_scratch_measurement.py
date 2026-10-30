from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from ai_measurement import AIJob, AIWorker
from measurement import Settings, centerline, summarize
from scratch_measurement import LocalScratchSegmenter, scratch_candidates
from test_app import FakeValue, headless_app, read_rows


class ScratchApplicationTests(unittest.TestCase):
    def test_loop_and_disconnected_loop_are_not_open_body_lines(self):
        mask = np.zeros((180, 300), np.uint8)
        cv2.circle(mask, (70, 90), 30, 1, 9)
        self.assertIsNone(centerline(mask)[0])
        cv2.line(mask, (170, 90), (260, 90), 1, 9)
        self.assertIsNone(centerline(mask)[0])
        mask[:] = 0
        cv2.line(mask, (40, 40), (100, 100), 1, 9)
        cv2.line(mask, (100, 100), (200, 100), 1, 9)
        self.assertIsNotNone(centerline(mask)[0])

    def test_app_submits_selected_backend_and_fine_button_does_not_switch_to_pretrained(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            app.ai_engine_choice = FakeValue(); app.ai_engine_choice.set('自作モデル（試験）')
            jobs = []
            class Worker:
                busy = False
                def submit(self, job):
                    jobs.append(job)
                    return True
            app.ai_failed = False; app.ai_worker = Worker()
            app.request_ai(app.frame, None)
            self.assertEqual(jobs[0].backend, 'scratch')
            self.assertIsNot(jobs[0].frame, app.frame)
            app.start_ai_fine()
            self.assertEqual(app.selected_ai_engine(), 'scratch')
            self.assertEqual(len(jobs), 1)

    def test_probability_geometry_restores_offsets_and_rejects_crossings(self):
        p = np.zeros((240, 340), np.float32)
        cv2.line(p, (60, 100), (260, 100), 1, 16)
        settings = Settings(appearance='dark', brightness=0)
        c = scratch_candidates(p, settings, (50, 40))
        self.assertEqual(summarize(c)['count'], 1)
        self.assertAlmostEqual(float(c[0].path[:, 1].mean()), 140, delta=1)
        cv2.line(p, (160, 40), (160, 160), 1, 16)
        self.assertEqual(summarize(scratch_candidates(p, settings))['count'], 0)

    def test_cut_and_invalid_probabilities_are_not_measurements(self):
        p = np.zeros((240, 340), np.float32)
        cv2.line(p, (0, 100), (200, 100), 1, 16)
        self.assertEqual(summarize(scratch_candidates(p, Settings()))['count'], 0)
        with self.assertRaises(ValueError):
            scratch_candidates(np.full((10, 10), np.nan), Settings())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'models'):
                LocalScratchSegmenter(Path(directory)/'absent.pt')

    def test_scratch_job_does_not_initialize_pretrained_engine(self):
        class Engine:
            def analyze(self, *_):
                return [], 'scratch-fake'

        def forbidden():
            raise AssertionError('Pretrained factory must not be used')

        worker = AIWorker(factory=forbidden, scratch_factory=Engine)
        try:
            job = AIJob(np.zeros((50, 50, 3), np.uint8), None, Settings(), None, 'token', backend='scratch')
            worker.submit(job)
            result = worker.results.get(timeout=3)
            self.assertIsNone(result['error'])
            self.assertEqual(result['device'], 'scratch-fake')
            self.assertIsNone(worker.engine)
        finally:
            worker.close()

    def test_switch_invalidates_inflight_result_and_clears_old_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            app.ai_engine_choice = FakeValue(); app.ai_engine_choice.set('自作モデル（試験）')
            old_job = AIJob(app.frame.copy(), None, app.settings, None, (app.series, app.ai_generation))
            app.result.set('old results')
            app.change_ai_engine()
            self.assertIn('まだ実行していません', app.result.get())
            self.assertTrue(app.paused)
            self.assertEqual(app.selected_ai_engine(), 'scratch')
            class Finished:
                def receive(self):
                    return {'job': old_job, 'error': None, 'candidates': [], 'device': 'old', 'seconds': 1}
            app.ai_worker = Finished(); app.ai_mode = 'single'
            app.poll_ai()
            self.assertFalse(app.csv_path.exists())
            self.assertIn('まだ実行していません', app.result.get())

    def test_scratch_result_is_logged_as_unverified_and_not_published_to_obs(self):
        with tempfile.TemporaryDirectory() as directory:
            app = headless_app(directory)
            job = AIJob(app.frame.copy(), None, app.settings, None, (app.series, app.ai_generation), backend='scratch')
            class Finished:
                def receive(self):
                    return {'job': job, 'error': None, 'candidates': [], 'device': 'cpu', 'seconds': 1}
            app.ai_mode, app.ai_worker, app.paused = 'single', Finished(), True
            app.poll_ai()
            self.assertIn('自作モデル', app.result.get())
            self.assertEqual(read_rows(app.csv_path)[0]['method'], 'scratch_auto_candidate')
            self.assertIn('体長の測定値なし', app.obs_path.read_text(encoding='utf-8-sig'))
