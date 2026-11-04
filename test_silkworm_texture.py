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

    def test_large_joined_regions_remain_visible_without_becoming_lengths(self):
        segmenter = LocalTexturePreviewSegmenter.__new__(LocalTexturePreviewSegmenter)
        segmenter.device = 'cpu'; segmenter.model = object()
        segmenter.metadata = {'inference_threshold': .65}
        frame = np.full((140, 230, 3), 128, np.uint8)
        probability = np.zeros((100, 170), np.float32)
        cv2.rectangle(probability, (15, 15), (150, 80), 1, -1)
        with patch('silkworm_scratch.predict', return_value=probability):
            candidates, _ = segmenter.analyze(frame, Settings(min_area=100, max_area=200), (20,15,190,115))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].reason, 'semantic_preview_large')
        self.assertIsNone(candidates[0].length_px)
        self.assertIsNone(candidates[0].path)
        self.assertEqual(segmenter.last_preview_mask.shape, (140, 230))
        self.assertEqual(segmenter.last_preview_mask[40,50], 255)
        self.assertEqual(segmenter.last_preview_mask[5,5], 0)
        self.assertTrue(np.all(segmenter.last_preview_mask[:, 190:] == 0))

    def test_preview_ui_has_no_false_counts_means_csv_or_obs_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            app = headless_app(folder)
            app.ai_engine_choice = FakeValue(); app.ai_engine_choice.set('自作輪郭（比較）')
            self.assertEqual(app.selected_ai_engine(), 'scratch_preview')
            app.candidates = [object(), object()]
            app.publish_candidates('scratch_region_preview', 'v6\n', True)
            self.assertIn('未確認の領域プレビュー', app.result.get())
            self.assertNotIn('2 領域', app.result.get())
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

    def test_previous_preview_model_can_be_compared_without_reloading_or_pretrained_weights(self):
        import time
        counts = {'new':0, 'old':0}
        class Engine:
            def __init__(self, key):
                counts[key] += 1
                self.metadata = {'model_version':key}
            def analyze(self, *_): return [], 'cpu'
        def forbidden(): raise AssertionError('Pretrained must not load')
        worker = AIWorker(factory=forbidden, texture_factory=lambda:Engine('new'),
                          texture_previous_factory=lambda:Engine('old'))
        try:
            for backend, expected in [('scratch_preview','new'),('scratch_preview_v6','old'),('scratch_preview','new')]:
                self.assertTrue(worker.submit(AIJob(np.zeros((40,50,3),np.uint8),None,Settings(),None,'t',backend=backend)))
                deadline = time.monotonic()+3
                result = worker.receive()
                while result is None and time.monotonic()<deadline:
                    time.sleep(.005); result = worker.receive()
                self.assertIsNotNone(result)
                self.assertIsNone(result['error'])
                self.assertEqual(result['model_version'], expected)
            self.assertEqual(counts, {'new':1,'old':1})
            self.assertIsNone(worker.engine)
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


    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch')
    def test_uniform_dimming_preserves_saturation_and_does_not_mutate_samples(self):
        import torch
        from train_silkworm_texture import dim_features
        source = torch.arange(72).reshape(2, 36).float()
        original = source.clone()
        result = dim_features(source, torch.tensor([[.7], [1.]]))
        unchanged = [3, 9, 14, 19, 24, 29, 34]
        self.assertTrue(torch.equal(result[:, unchanged], source[:, unchanged]))
        self.assertTrue(torch.equal(source, original))
        self.assertAlmostEqual(float(result[0, 5]), 3.5)
        self.assertTrue(torch.equal(result[1], source[1]))
        with self.assertRaises(ValueError): dim_features(source, torch.tensor([[-.1], [1.]]))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch')
    def test_tiled_features_match_whole_image_across_non_square_seams(self):
        import torch
        from silkworm_texture import create_texture_model, predict_texture
        torch.set_num_threads(2); torch.manual_seed(13)
        model = create_texture_model().eval()
        image = np.random.default_rng(8).integers(0, 256, (317, 389, 3), dtype=np.uint8)
        whole = predict_texture(model, image, tile_size=None)
        tiled = predict_texture(model, image, batch_size=1024, tile_size=128)
        self.assertEqual(tiled.shape, (317, 389))
        np.testing.assert_allclose(tiled, whole, atol=1e-5)
        with self.assertRaises(ValueError): predict_texture(model, image, tile_size=0)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional torch')
    def test_scaled_training_keeps_unknown_labels_and_rejects_validation_overlap(self):
        import hashlib
        from PIL import Image
        from train_silkworm_texture import train
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (64, 80), 'gray').save(root/'source.png')
            mask = np.full((80, 64), 255, np.uint8)
            mask[10:20, 10:25] = 1; mask[45:60, 35:55] = 0
            Image.fromarray(mask).save(root/'labels.png')
            digest = hashlib.sha256((root/'source.png').read_bytes()).hexdigest()
            dataset = {'ignore_value':255, 'frames':[{'id':'tiny', 'image':'source.png',
                       'labels':'labels.png', 'image_sha256':digest, 'roi':[0,0,64,80]}]}
            (root/'dataset.json').write_text(json.dumps(dataset))
            with self.assertRaisesRegex(ValueError, '64 hexadecimal'):
                train(root/'dataset.json', root/'invalid.pt', steps=1, validation_image_sha256='not-a-sha')
            with self.assertRaisesRegex(ValueError, 'Validation image'):
                train(root/'dataset.json', root/'blocked.pt', steps=1, validation_image_sha256=digest)
            self.assertFalse((root/'blocked.pt').exists())
            info = train(root/'dataset.json', root/'own.pt', steps=1,
                         source_scales=(.75, 1., 1.35), augment_exposure=True, per_class_per_frame=30)
            self.assertEqual(info['augmentation']['source_scales'], [.75, 1., 1.35])
            self.assertEqual(info['initialization'], 'random_no_pretrained_weights')
            np.testing.assert_array_equal(np.asarray(Image.open(root/'labels.png')), mask)


if __name__ == '__main__': unittest.main()
