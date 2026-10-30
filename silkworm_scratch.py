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


def create_model(architecture='kaiko_small_unet_v1'):
    import torch
    from torch import nn
    from torch.nn import functional as F

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.LeakyReLU(.1),
                             nn.Conv2d(b, b, 3, padding=1), nn.LeakyReLU(.1))

    if architecture not in ('kaiko_small_unet_v1', 'kaiko_context_unet_v2'):
        raise ValueError('Unsupported architecture')

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = block(3, 8)
            self.enc2 = block(8, 16)
            if architecture == 'kaiko_context_unet_v2':
                self.enc3 = block(16, 24)
                self.bottom = block(24, 32)
                self.dec3 = block(56, 24)
                self.dec2 = block(40, 16)
            else:
                self.bottom = block(16, 32)
                self.dec2 = block(48, 16)
            self.dec1 = block(24, 8)
            self.output = nn.Conv2d(8, 1, 1)

        def forward(self, x):
            a = self.enc1(x)
            b = self.enc2(F.max_pool2d(a, 2))
            if architecture == 'kaiko_context_unet_v2':
                c = self.enc3(F.max_pool2d(b, 2))
                z = self.bottom(F.max_pool2d(c, 2))
                c = self.dec3(torch.cat([F.interpolate(z, size=c.shape[-2:], mode='bilinear', align_corners=False), c], 1))
            else:
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
        if item.get('labels_sha256') and hashlib.sha256(paths[1].read_bytes()).hexdigest() != item['labels_sha256']:
            raise ValueError('Label hash mismatch')
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


def train(dataset, output, steps=1000, device='auto', seed=41,
          architecture='kaiko_small_unet_v1', patch_size=128, batch_size=4, resume=None):
    import torch
    if steps < 1:
        raise ValueError('Steps must be positive')
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    config, frames = load_dataset(dataset)
    if patch_size < 8 or patch_size % 8 or batch_size < 2:
        raise ValueError('Patch size must be a multiple of eight; batch size must be at least two')
    if resume:
        model, previous = load_model(resume)
        if previous['architecture'] != architecture:
            raise ValueError('Resume architecture differs')
    else:
        model = create_model(architecture)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0002 if resume else (.0006 if architecture == 'kaiko_context_unet_v2' else .001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=.00001)
    history, recent = [], []
    start = time.perf_counter()
    for step in range(1, steps+1):
        x, y = sample_batch(frames, rng, count=batch_size, size=patch_size)
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
    metadata = {'schema_version': 1, 'architecture': architecture,
                'initialization': 'checkpoint_resume' if resume else 'random_no_pretrained_weights', 'seed': seed,
                'parent_checkpoint_sha256': hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else None,
                'training_steps': steps, 'training_images': len(frames), 'device': device,
                'learning_rate_schedule': 'cosine_to_0.00001', 'patch_size': patch_size, 'batch_size': batch_size,
                'parameters': sum(p.numel() for p in model.parameters()),
                'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
                'training_sources': [{'id': f.get('id'), 'image_sha256': f['image_sha256'],
                                      'labels_sha256': hashlib.sha256((Path(dataset).parent/f['labels']).read_bytes()).hexdigest()}
                                     for f in config['frames']],
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
    model = create_model(checkpoint['metadata']['architecture'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint['metadata']


def predict(model, rgb, tile=384, halo=64):
    import torch
    if rgb.ndim != 3 or rgb.shape[2] != 3 or tile < 4 or tile % 4 or halo % 4:
        raise ValueError('Expected RGB image, and tile/halo multiples of four')
    h, w = rgb.shape[:2]
    # Same padding at image edges; copy only tile interiors to reduce seams.
    padded = np.pad(rgb, ((halo, halo), (halo, halo), (0, 0)), mode='edge')
    output = np.empty((h, w), np.float32)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = 'cpu'
    with torch.inference_mode():
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                crop = padded[y:min(y+tile, h)+2*halo, x:min(x+tile, w)+2*halo]
                # Align the encoder's three downsampling stages without changing
                # any source pixels or stretching the final tile.
                crop = np.pad(crop, ((0, (-crop.shape[0]) % 8),
                                     (0, (-crop.shape[1]) % 8), (0, 0)), mode='edge')
                tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)[None])).float()/255
                probabilities = model(tensor.to(device)).sigmoid()[0, 0].cpu().numpy()
                bh, bw = min(tile, h-y), min(tile, w-x)
                output[y:y+bh, x:x+bw] = probabilities[halo:halo+bh, halo:halo+bw]
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--architecture', default='kaiko_small_unet_v1', choices=['kaiko_small_unet_v1', 'kaiko_context_unet_v2'])
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume', help='Continue training from a supported checkpoint')
    args = parser.parse_args()
    train(args.dataset, args.output, args.steps, args.device,
          architecture=args.architecture, patch_size=args.patch_size, batch_size=args.batch_size, resume=args.resume)
