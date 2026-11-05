"""Write a static, unverified region preview from our experimental model."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from silkworm_scratch import load_model, predict


def preview(model_path, image_path, output, roi=None, threshold=.5):
    import torch
    if not 0 < threshold < 1:
        raise ValueError('Threshold must be between zero and one')
    torch.set_num_threads(3)
    with Image.open(image_path) as source:
        rgb = np.asarray(source.convert('RGB'))
    h, w = rgb.shape[:2]
    roi = [0, 0, w, h] if roi is None else list(roi)
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('ROI must stay inside the image')
    crop = rgb[y1:y2, x1:x2]
    model, metadata = load_model(model_path)
    start = time.perf_counter()
    probabilities = predict(model, crop)
    elapsed = time.perf_counter()-start
    mask = (probabilities >= threshold).astype(np.uint8)
    overlay = crop.copy()
    overlay[mask > 0] = (.7*crop[mask > 0]+.3*np.array([0, 255, 80])).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 170, 0), 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(output/'input.png')
    Image.fromarray(overlay).save(output/'unverified_regions.png')
    Image.fromarray((probabilities*255).astype(np.uint8)).save(output/'probability.png')
    report = {'architecture': metadata['architecture'], 'roi': roi, 'threshold': threshold,
              'cpu_inference_seconds': elapsed,
              'status': 'unverified_semantic_regions_not_counts_or_lengths',
              'ground_truth_evaluation': False}
    (output/'preview.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'))
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    print(json.dumps(preview(args.model, args.image, args.output, args.roi, args.threshold)))
