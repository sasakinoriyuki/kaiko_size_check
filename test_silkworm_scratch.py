import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from silkworm_scratch import create_model, load_dataset, masked_loss, predict, sample_batch


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional training dependency is not installed')
class ScratchTrainingTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2)

    def test_unknown_pixels_never_affect_loss_or_gradients(self):
        import torch
        truth = torch.tensor([[[[0, 1, 255]]]])
        logits = torch.tensor([[[[.3, -.2, 100.]]]], requires_grad=True)
        loss = masked_loss(logits, truth)
        loss.backward()
        self.assertEqual(logits.grad[0, 0, 0, 2], 0)
        changed = logits.detach().clone(); changed[0, 0, 0, 2] = -100
        self.assertEqual(float(loss.detach()), float(masked_loss(changed, truth)))
        with self.assertRaises(ValueError):
            masked_loss(logits, torch.full_like(truth, 255))
        with self.assertRaises(ValueError):
            masked_loss(logits, torch.full_like(truth, 2))

    def test_model_preserves_non_square_odd_image_dimensions(self):
        import torch
        model = create_model()
        self.assertLess(sum(p.numel() for p in model.parameters()), 40000)
        with torch.inference_mode():
            out = model(torch.zeros(1, 3, 67, 93))
        self.assertEqual(tuple(out.shape), (1, 1, 67, 93))

    def test_sampling_keeps_unknown_labels_and_only_discrete_classes(self):
        image = np.zeros((160, 160, 3), np.uint8)
        labels = np.full((160, 160), 255, np.uint8)
        labels[25:45, 25:45] = 1
        labels[100:120, 100:120] = 0
        centres = [np.argwhere(labels == cls) for cls in (0, 1)]
        x, y = sample_batch([(image, labels, centres)], np.random.default_rng(1))
        self.assertEqual(x.shape, (4, 3, 128, 128))
        self.assertEqual(set(np.unique(y)), {0, 1, 255})
        self.assertTrue(x.flags.c_contiguous and y.flags.c_contiguous)
        self.assertEqual(set(np.unique(labels)), {0, 1, 255})

    def test_dataset_rejects_changed_images_and_background_outside_roi(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new('RGB', (160, 160), 'green').save(root/'image.png')
            labels = np.full((160, 160), 255, np.uint8)
            labels[40:50, 40:50] = 1; labels[80:90, 80:90] = 0
            Image.fromarray(labels).save(root/'labels.png')
            item = {'image': 'image.png', 'labels': 'labels.png',
                    'image_sha256': hashlib.sha256((root/'image.png').read_bytes()).hexdigest(),
                    'roi': [20, 20, 140, 140]}
            data = {'ignore_value': 255, 'frames': [item]}
            config = root/'dataset.json'; config.write_text(json.dumps(data))
            self.assertEqual(len(load_dataset(config)[1]), 1)
            labels[0, 0] = 0; Image.fromarray(labels).save(root/'labels.png')
            with self.assertRaisesRegex(ValueError, 'outside ROI'):
                load_dataset(config)
            Image.new('RGB', (160, 160), 'white').save(root/'image.png')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_dataset(config)

    def test_dense_prediction_covers_edges_without_resizing(self):
        import torch
        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
        result = predict(ConstantModel(), np.zeros((91, 137, 3), np.uint8), tile=64, halo=16)
        self.assertEqual(result.shape, (91, 137))
        np.testing.assert_array_equal(result, np.full((91, 137), .5))


if __name__ == '__main__':
    unittest.main()
