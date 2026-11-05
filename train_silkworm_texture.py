"""Train the own texture-region preview model from explicit sparse pixel labels."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import time

import cv2
import numpy as np


def train(dataset, output, steps=2200, seed=42, per_class_per_frame=20000,
          source_scales=(1.,), augment_exposure=False, model_version="own texture preview",
          validation_image_sha256=None):
    import torch
    from silkworm_scratch import load_dataset
    from silkworm_texture import create_texture_model, features, ARCHITECTURE
    if steps < 1 or per_class_per_frame < 1:
        raise ValueError('Training/sample counts must be positive')
    if not source_scales or any(not np.isfinite(s) or not .5 <= s <= 2 for s in source_scales):
        raise ValueError('Source scales must be between .5 and 2')
    if validation_image_sha256 is not None:
        if not isinstance(validation_image_sha256, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', validation_image_sha256):
            raise ValueError('Validation image SHA256 must contain 64 hexadecimal characters')
        validation_image_sha256 = validation_image_sha256.lower()
    torch.set_num_threads(3)
    dataset = Path(dataset)
    # Freeze provenance at load time; never read changed data to describe training.
    dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config, frames = load_dataset(dataset)
    provenance = [{'id': f.get('id'), 'image_sha256': f['image_sha256'],
                   'labels_sha256': hashlib.sha256((dataset.parent/f['labels']).read_bytes()).hexdigest()}
                  for f in config['frames']]
    if validation_image_sha256 and validation_image_sha256 in [f['image_sha256'] for f in config['frames']]:
        raise ValueError('Validation image is present in the training dataset')
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for image, labels, _ in frames:
        for scale in source_scales:
            if scale == 1:
                rgb, target = image, labels
            else:
                rgb = cv2.resize(image, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                target = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            f = features(rgb)
            for cls in (0, 1):
                pool = np.argwhere(target == cls)
                if not len(pool):
                    raise ValueError('Scaled image lost a labelled class')
                selected = pool[rng.choice(len(pool), min(per_class_per_frame, len(pool)), replace=False)]
                inputs.append(f[selected[:, 0], selected[:, 1]])
                targets.append(np.full(len(selected), cls, np.float32))
            del f
    x = np.concatenate(inputs)
    mean, std = x.mean(0), x.std(0).clip(.02)
    x = torch.from_numpy(x)
    y = torch.from_numpy(np.concatenate(targets)[:, None])
    torch.manual_seed(seed)
    model = create_texture_model()
    model.feature_mean.copy_(torch.from_numpy(mean))
    model.feature_std.copy_(torch.from_numpy(std))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(seed)
    history, started = [], time.monotonic()
    for step in range(steps):
        index = torch.randint(len(x), (2048,), generator=generator)
        batch = x[index]
        if augment_exposure:
            # These features transform linearly under uniform dimming without
            # clipping. Saturation and its local means remain unchanged.
            gains = .6+.4*torch.rand((len(batch), 1), generator=generator)
            gains[torch.rand((len(batch), 1), generator=generator) < .25] = 1
            batch = dim_features(batch, gains)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), y[index])
        if not torch.isfinite(loss): raise ValueError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 200 == 0 or step == steps-1:
            row = {'step': step+1, 'training_loss': float(loss.detach()),
                   'elapsed_seconds': round(time.monotonic()-started, 2)}
            history.append(row)
            print(json.dumps(row), flush=True)
    metadata = {'schema_version': 1, 'architecture': ARCHITECTURE,
                'model_version': model_version, 'seed': seed,
                'initialization': 'random_no_pretrained_weights',
                'training_steps': steps, 'training_images': len(frames),
                'sampling_seed': seed, 'per_class_per_frame': per_class_per_frame,
                'augmentation': {'source_scales': list(source_scales),
                                 'uniform_exposure_gain': [.6, 1.] if augment_exposure else None,
                                 'exposure_unchanged_fraction': .25 if augment_exposure else 1.},
                'validation_image_not_used_in_training': validation_image_sha256,
                'dataset_sha256': dataset_digest, 'training_sources': provenance,
                'inference_threshold': .65, 'history': history,
                'parameters': sum(p.numel() for p in model.parameters()),
                'purpose': 'semantic_region_preview_only_no_counts_no_lengths',
                'independent_evaluation': 'not_performed'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'metadata': metadata, 'state_dict': model.state_dict()}, output)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def dim_features(batch, gains):
    """Uniform gain preserves HSV saturation, while intensity/texture scale."""
    import torch
    if batch.ndim != 2 or batch.shape[1] != 36 or gains.shape != (len(batch), 1):
        raise ValueError('Expected 36 features and one gain per sample')
    if not torch.isfinite(gains).all() or torch.any(gains <= 0) or torch.any(gains > 1):
        raise ValueError('Dimming gains must be finite and in (0, 1]')
    result = batch.clone()
    indices = [j for j in range(36) if j not in (3, 9, 14, 19, 24, 29, 34)]
    result[:, indices] = result[:, indices]*gains
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=2200)
    parser.add_argument('--scales', type=float, nargs='+', default=[1.])
    parser.add_argument('--augment-exposure', action='store_true')
    parser.add_argument('--samples-per-class', type=int, default=20000)
    parser.add_argument('--model-version', default='own texture preview')
    parser.add_argument('--validation-image-sha256')
    args = parser.parse_args()
    train(args.dataset, args.output, args.steps, per_class_per_frame=args.samples_per_class,
          source_scales=args.scales, augment_exposure=args.augment_exposure,
          model_version=args.model_version, validation_image_sha256=args.validation_image_sha256)
