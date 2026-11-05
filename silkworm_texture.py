"""Own pixel classifier using colour and local texture; no pretrained weights.

The model produces semantic region previews, never individual counts or lengths.
The feature order is part of the checkpoint architecture, not a learned detector.
"""
import cv2
import numpy as np

ARCHITECTURE = 'kaiko_texture_mlp_v1'
FEATURE_COUNT = 36


def features(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError('Expected non-empty uint8 RGB image')
    a = rgb.astype(np.float32)/255
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    sat = hsv[:, :, 1]
    values = [a[:, :, i] for i in range(3)] + [sat, a.max(2), gray]
    for sigma in (1, 2, 4, 8, 16, 32):
        mean = cv2.GaussianBlur(gray, (0, 0), sigma)
        variance = np.maximum(0, cv2.GaussianBlur(gray*gray, (0, 0), sigma)-mean*mean)
        dx = cv2.Sobel(mean, cv2.CV_32F, 1, 0, ksize=3)*sigma
        dy = cv2.Sobel(mean, cv2.CV_32F, 0, 1, ksize=3)*sigma
        values.extend((mean, np.sqrt(variance), 2*(gray-mean),
                       cv2.GaussianBlur(sat, (0, 0), sigma), np.sqrt(dx*dx+dy*dy)))
    return np.stack(values, axis=2)


def create_texture_model():
    import torch
    from torch import nn

    class TextureMLP(nn.Module):
        feature_input = True

        def __init__(self):
            super().__init__()
            self.register_buffer('feature_mean', torch.zeros(FEATURE_COUNT))
            self.register_buffer('feature_std', torch.ones(FEATURE_COUNT))
            self.layers = nn.Sequential(nn.Linear(FEATURE_COUNT, 64), nn.LeakyReLU(.1),
                                        nn.Linear(64, 32), nn.LeakyReLU(.1), nn.Linear(32, 1))

        def forward(self, x):
            return self.layers((x-self.feature_mean)/self.feature_std.clamp_min(.02))

    return TextureMLP()


def predict_texture(model, rgb, batch_size=32768, tile_size=768):
    """Preserve source pixels and use overlapping tiles to bound feature memory.

    136 pixels cover the largest Gaussian radius (128), plus its Sobel
    derivative. Border handling therefore matches a whole-image calculation.
    Pass tile_size=None only for reference comparisons on bounded images.
    """
    import torch
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError('Expected non-empty uint8 RGB image')
    if batch_size < 1 or (tile_size is not None and (not isinstance(tile_size, int) or tile_size < 1)):
        raise ValueError('Batch/tile sizes must be positive')
    height, width = rgb.shape[:2]
    step = max(height, width) if tile_size is None else tile_size
    probability = np.empty((height, width), np.float32)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for top in range(0, height, step):
            for left in range(0, width, step):
                bottom, right = min(top+step, height), min(left+step, width)
                y0, x0 = max(0, top-136), max(0, left-136)
                y1, x1 = min(height, bottom+136), min(width, right+136)
                f = features(rgb[y0:y1, x0:x1])
                inner = f[top-y0:bottom-y0, left-x0:right-x0]
                flat = np.ascontiguousarray(inner).reshape(-1, FEATURE_COUNT)
                result = np.empty(len(flat), np.float32)
                for start in range(0, len(flat), batch_size):
                    tensor = torch.from_numpy(flat[start:start+batch_size]).to(device)
                    result[start:start+batch_size] = model(tensor).sigmoid()[:, 0].cpu().numpy()
                probability[top:bottom, left:right] = result.reshape(bottom-top, right-left)
    return probability
