"""Train the own texture-region preview model from explicit sparse pixel labels."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def train(dataset, output, steps=2200, seed=42, per_class_per_frame=20000):
    import torch
    from silkworm_scratch import load_dataset
    from silkworm_texture import create_texture_model, features, ARCHITECTURE
    if steps < 1 or per_class_per_frame < 1:
        raise ValueError('Training/sample counts must be positive')
    torch.set_num_threads(3)
    dataset = Path(dataset)
    # Freeze provenance at load time; never read changed data to describe training.
    dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config, frames = load_dataset(dataset)
    provenance = [{'id': f.get('id'), 'image_sha256': f['image_sha256'],
                   'labels_sha256': hashlib.sha256((dataset.parent/f['labels']).read_bytes()).hexdigest()}
                  for f in config['frames']]
    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    for image, _, centres in frames:
        f = features(image)
        for cls in (0, 1):
            pool = centres[cls]
            selected = pool[rng.choice(len(pool), min(per_class_per_frame, len(pool)), replace=False)]
            inputs.append(f[selected[:, 0], selected[:, 1]])
            targets.append(np.full(len(selected), cls, np.float32))
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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[index]), y[index])
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
                'model_version': 'own texture preview', 'seed': seed,
                'initialization': 'random_no_pretrained_weights',
                'training_steps': steps, 'training_images': len(frames),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=2200)
    args = parser.parse_args()
    train(args.dataset, args.output, args.steps)
