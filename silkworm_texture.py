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


def predict_texture(model, rgb, batch_size=32768):
    """Compute features at source resolution; bound device-side tensor memory."""
    import torch
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    f = features(rgb)
    shape = f.shape[:2]
    flat = f.reshape(-1, FEATURE_COUNT)
    probability = np.empty(len(flat), np.float32)
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            tensor = torch.from_numpy(flat[start:start+batch_size]).to(device)
            probability[start:start+batch_size] = model(tensor).sigmoid()[:, 0].cpu().numpy()
    return probability.reshape(shape)
