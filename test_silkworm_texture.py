import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from ai_measurement import AIJob, AIWorker
from measurement import Settings
from scratch_measurement import LocalTexturePreviewSegmenter
from silkworm_texture import features, FEATURE_COUNT
from test_app import headless_app, FakeValue


class TexturePreviewTest(unittest.TestCase):
    def test_features_require_original_rgb_and_preserve_odd_coordinates(self):
        f = features(np.full((51, 73, 3), 128, np.uint8))
        self.assertEqual(f.shape, (51, 73, FEATURE_COUNT))
        self.assertTrue(np.isfinite(f).all())
        np.testing.assert_allclose(f[:, :, :3], 128/255)
        for bad in [np.zeros((0, 3, 3), np.uint8), np.zeros((5, 7), np.uint8), np.zeros((5, 7, 3), np.float32)]:
            with self.assertRaises(ValueError): features(bad)

    def test_preview_mask_never_exports_a_length_even_for_an_ideal_body(self):
        segmenter = LocalTexturePreviewSegmenter.__new__(LocalTexturePreviewSegmenter)
        segmenter.device = 'cpu'; segmenter.model = object()
        segmenter.metadata = {'inference_threshold': .65}
        probability = np.zeros((100, 170), np.float32)
        cv2.line(probability, (30, 50), (140, 50), 1, 12)
        frame = np.full((140, 230, 3), 128, np.uint8)
        with patch('silkworm_scratch.predict', return_value=probability):
            candidates, device = segmenter.analyze(frame, Settings(), (20, 15, 190, 115))
        self.assertEqual(len(candidates), 1)
        self.assertTrue(all(c.path is None and c.length_px is None and c.reason == 'semantic_preview' for c in candidates))
        self.assertEqual(int(candidates[0].contour[:, :, 0].min()), 44)
        self.assertEqual(device, 'cpu')

    def test_preview_ui_has_no_false_counts_means_csv_or_obs_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            app = headless_app(folder)
            app.ai_engine_choice = FakeValue(); app.ai_engine_choice.set('自作輪郭（比較v6）')
            self.assertEqual(app.selected_ai_engine(), 'scratch_preview')
            app.candidates = [object(), object()]
            app.publish_candidates('scratch_region_preview', 'v6\n', True)
            self.assertIn('輪郭候補 2 領域', app.result.get())
            self.assertNotIn('平均長', app.result.get())
            self.assertIn('頭数・自動体長は計測しません', app.result.get())
            self.assertFalse(app.csv_path.exists())
            self.assertEqual(app.history, [])
            self.assertIn('体長の測定値なし', app.obs_path.read_text(encoding='utf-8-sig'))
            previous = app.mode
            app.begin('ai_point')
            self.assertEqual(app.mode, previous)
            self.assertIsNone(app.ai_worker)
            app.start_ai_fine()
            self.assertEqual(app.selected_ai_engine(), 'scratch_preview')

    def test_preview_weights_cannot_be_routed_to_length_extraction(self):
        from scratch_measurement import LocalScratchSegmenter
        segmenter = LocalScratchSegmenter.__new__(LocalScratchSegmenter)
        segmenter.metadata = {'purpose': 'semantic_region_preview_only_no_counts_no_lengths'}
        with self.assertRaisesRegex(RuntimeError, '輪郭確認モデル'):
            segmenter.analyze(np.zeros((50, 60, 3), np.uint8), Settings())

    def test_worker_preview_keeps_both_other_models_unloaded(self):
        class Preview:
            metadata = {'model_version': 'v6 輪郭比較', 'inference_threshold': .65}
            def analyze(self, *_): return [], 'cpu'
        def forbidden(): raise AssertionError('Other weights must not load')
        worker = AIWorker(factory=forbidden, scratch_factory=forbidden, texture_factory=Preview)
        try:
            worker.submit(AIJob(np.zeros((40, 50, 3), np.uint8), None, Settings(), None, 't', backend='scratch_preview'))
            result = worker.results.get(timeout=3)
            self.assertIsNone(result['error'])
            self.assertIn('v6', result['model_version'])
            self.assertIsNone(worker.engine)
            self.assertIsNone(worker.scratch_engine)
        finally: worker.close()

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch')
    def test_feature_checkpoint_roundtrip_is_finite_and_preserves_shape(self):
        import torch
        from silkworm_scratch import create_model, load_model, predict
        torch.set_num_threads(2)
        model = create_model('kaiko_texture_mlp_v1').eval()
        with torch.no_grad():
            for parameter in model.parameters(): parameter.zero_()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'own.pt'
            torch.save({'metadata': {'architecture': 'kaiko_texture_mlp_v1'}, 'state_dict': model.state_dict()}, path)
            restored, _ = load_model(path)
            p = predict(restored, np.full((51, 73, 3), 255, np.uint8))
        np.testing.assert_array_equal(p, np.full((51, 73), .5))


if __name__ == '__main__': unittest.main()
