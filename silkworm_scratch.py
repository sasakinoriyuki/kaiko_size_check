"""Small segmentation experiment trained from random initialization.

Torch is imported only when a training/inference function is called.
Labels: 0 = explicitly annotated background, 1 = body, 255 = unknown.
This module does not identify individual animals or measure their length.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image


def create_model():
    import torch
    from torch import nn
    from torch.nn import functional as F

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.LeakyReLU(.1),
                             nn.Conv2d(b, b, 3, padding=1), nn.LeakyReLU(.1))

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = block(3, 8)
            self.enc2 = block(8, 16)
            self.bottom = block(16, 32)
            self.dec2 = block(48, 16)
            self.dec1 = block(24, 8)
            self.output = nn.Conv2d(8, 1, 1)

        def forward(self, x):
            a = self.enc1(x)
            b = self.enc2(F.max_pool2d(a, 2))
            c = self.bottom(F.max_pool2d(b, 2))
            c = self.dec2(torch.cat([F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1))
            c = self.dec1(torch.cat([F.interpolate(c, size=a.shape[-2:], mode='bilinear', align_corners=False), a], 1))
            return self.output(c)

    return SmallUNet()


def masked_loss(logits, labels):
    import torch
    from torch.nn import functional as F
    if logits.shape != labels.shape:
        raise ValueError('Label and prediction shapes differ')
    if not torch.all((labels == 0) | (labels == 1) | (labels == 255)):
        raise ValueError('Labels must be 0, 1 or 255')
    valid = labels != 255
    if not torch.any(valid):
        raise ValueError('No labelled pixels')
    # Average the two classes separately, so sparse positive labels cannot be
    # drowned out by larger background rectangles. Unknown pixels have no loss.
    loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float(), reduction='none')
    truth = labels[valid]
    parts = [loss[truth == cls].mean() for cls in (0, 1) if torch.any(truth == cls)]
    return torch.stack(parts).mean()


def load_dataset(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('ignore_value') != 255 or not config.get('frames'):
        raise ValueError('Expected sparse dataset with ignore_value 255')
    frames = []
    for item in config['frames']:
        paths = [(path.parent / item[key]).resolve() for key in ('image', 'labels')]
        if any(not p.is_relative_to(path.parent.resolve()) for p in paths):
            raise ValueError('Dataset path leaves its folder')
        if hashlib.sha256(paths[0].read_bytes()).hexdigest() != item['image_sha256']:
            raise ValueError('Image hash mismatch')
        image = np.asarray(Image.open(paths[0]).convert('RGB'))
        labels = np.asarray(Image.open(paths[1]))
        if image.shape[:2] != labels.shape or not np.isin(labels, [0, 1, 255]).all():
            raise ValueError('Invalid label image')
        x1, y1, x2, y2 = item['roi']
        if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
            raise ValueError('Invalid ROI')
        # Refuse labels outside the camera view, e.g. the OBS sidebar.
        valid = labels != 255
        valid[y1:y2, x1:x2] = False
        if valid.any():
            raise ValueError('Labels outside ROI')
        image = image[y1:y2, x1:x2].copy()
        labels = labels[y1:y2, x1:x2].copy()
        centres = [np.argwhere(labels == cls) for cls in (0, 1)]
        if any(len(c) == 0 for c in centres):
            raise ValueError('Each frame needs explicit body and background labels')
        frames.append((image, labels, centres))
    return config, frames


def sample_batch(frames, rng, count=4, size=128):
    images, labels = [], []
    for i in range(count):
        image, mask, centres = frames[int(rng.integers(len(frames)))]
        if min(image.shape[:2]) < size:
            raise ValueError('Training ROI is smaller than the requested patch size')
        pool = centres[i % 2]
        y, x = pool[int(rng.integers(len(pool)))]
        left = int(np.clip(x-size//2+rng.integers(-24, 25), 0, image.shape[1]-size))
        top = int(np.clip(y-size//2+rng.integers(-24, 25), 0, image.shape[0]-size))
        crop = image[top:top+size, left:left+size].astype(np.float32)/255
        target = mask[top:top+size, left:left+size].copy()
        rotation = int(rng.integers(4))
        crop, target = np.rot90(crop, rotation), np.rot90(target, rotation)
        if rng.random() < .5:
            crop, target = crop[:, ::-1], target[:, ::-1]
        crop = np.clip(crop*rng.uniform(.85, 1.15)+rng.uniform(-.04, .04), 0, 1)
        images.append(crop.transpose(2, 0, 1))
        labels.append(target[None])
    return np.ascontiguousarray(images, dtype=np.float32), np.ascontiguousarray(labels)


def train(dataset, output, steps=1000, device='auto', seed=41):
    import torch
    if steps < 1:
        raise ValueError('Steps must be positive')
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    config, frames = load_dataset(dataset)
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=.00001)
    history, recent = [], []
    start = time.perf_counter()
    for step in range(1, steps+1):
        x, y = sample_batch(frames, rng)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(model(torch.from_numpy(x).to(device)), torch.from_numpy(y).to(device))
        if not torch.isfinite(loss):
            raise ValueError('Training loss became non-finite')
        loss.backward()
        optimizer.step()
        scheduler.step()
        recent.append(float(loss.detach().cpu()))
        if step == 1 or step % 100 == 0 or step == steps:
            row = {'step': step, 'mean_training_loss': float(np.mean(recent)),
                   'elapsed_seconds': round(time.perf_counter()-start, 2)}
            history.append(row)
            print(json.dumps(row), flush=True)
            recent = []
    metadata = {'schema_version': 1, 'architecture': 'kaiko_small_unet_v1',
                'initialization': 'random_no_pretrained_weights', 'seed': seed,
                'training_steps': steps, 'training_images': len(frames), 'device': device,
                'learning_rate_schedule': 'cosine_0.001_to_0.00001',
                'parameters': sum(p.numel() for p in model.parameters()),
                'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                'annotation_review_sha256': config.get('annotation_review_sha256'),
                'history': history, 'independent_evaluation': 'not_performed',
                'use': 'unverified_semantic_body_regions_not_counts_or_lengths'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'metadata': metadata, 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}}, output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def load_model(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['metadata']['architecture'] != 'kaiko_small_unet_v1':
        raise ValueError('Unsupported architecture')
    model = create_model()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint['metadata']


def predict(model, rgb, tile=384, halo=32):
    import torch
    if rgb.ndim != 3 or rgb.shape[2] != 3 or tile < 4 or tile % 4 or halo % 4:
        raise ValueError('Expected RGB image, and tile/halo multiples of four')
    h, w = rgb.shape[:2]
    # Same padding at image edges; copy only tile interiors to reduce seams.
    padded = np.pad(rgb, ((halo, halo), (halo, halo), (0, 0)), mode='edge')
    output = np.empty((h, w), np.float32)
    with torch.inference_mode():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                crop = padded[y:min(y+tile, h)+2*halo, x:min(x+tile, w)+2*halo]
                tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)[None])).float()/255
                probabilities = model(tensor).sigmoid()[0, 0].numpy()
                bh, bw = min(tile, h-y), min(tile, w-x)
                output[y:y+bh, x:x+bw] = probabilities[halo:halo+bh, halo:halo+bw]
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    train(args.dataset, args.output, args.steps, args.device)
