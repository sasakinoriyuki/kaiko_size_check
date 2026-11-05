"""Optional local inference with the channel's own model; identity is unverified."""
from pathlib import Path
import os
import hashlib

import cv2
import numpy as np

from ai_measurement import content_view, mask_geometry
from measurement import Candidate


def scratch_candidates(probabilities, settings, offset=(0, 0), threshold=.5):
    settings.validate()
    p = np.asarray(probabilities)
    if p.ndim != 2 or not np.isfinite(p).all() or not 0 < threshold < 1:
        raise ValueError('Invalid model probabilities')
    mask = (p >= threshold).astype(np.uint8)
    # Closing only fills tiny breaks; do not bridge large occlusions or split
    # touching animals into invented complete bodies.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    for i in range(1, count):
        if not settings.min_area <= stats[i, cv2.CC_STAT_AREA] <= settings.max_area:
            continue
        x, y, w, h = map(int, stats[i, :4])
        local = np.pad((labels[y:y+h, x:x+w] == i).astype(np.uint8), 3)
        # Predicted contours can have tiny serrations. Reuse bounded spur
        # cleanup (no longer than the local radius); long branches, loops,
        # disconnected pieces and edge-cut bodies are still rejected.
        proposal = mask_geometry(local, settings, (offset[0]+x-3, offset[1]+y-3), guided=True)
        if proposal is not None:
            if x <= 2 or y <= 2 or x+w >= mask.shape[1]-2 or y+h >= mask.shape[0]-2:
                c = proposal['candidate']
                c.path, c.length_px, c.reason = None, None, 'edge'
            candidates.append(proposal['candidate'])
    return candidates


class LocalScratchSegmenter:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else Path(__file__).parent/'models/kaiko_scratch.pt'
        if not path.is_file():
            raise RuntimeError('自作モデルがありません。配布ZIPのmodelsフォルダもアプリと同じ場所へコピーしてください。')
        try:
            import torch
            from silkworm_scratch import load_model
        except ImportError as exc:
            hint = '蚕チェック.bat' if os.environ.get('KAIKO_ONE_CLICK') == '1' else 'setup_scratch_windows.bat'
            raise RuntimeError(f'自作モデル用のライブラリを読み込めません。{hint}を開いてください。') from exc
        torch.set_num_threads(3)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.model, self.metadata = load_model(path)
        self.model.to(self.device).eval()

    def analyze(self, frame, settings, roi=None):
        if self.metadata.get('purpose') == 'semantic_region_preview_only_no_counts_no_lengths':
            raise RuntimeError('輪郭確認モデルは自動体長に使えません。自作輪郭の解析方式を選んでください。')
        from silkworm_scratch import predict
        view, offset = content_view(frame, roi)
        if view is None:
            return [], self.device
        probability = predict(self.model, cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        return scratch_candidates(probability, settings, offset,
                                  threshold=float(self.metadata.get('inference_threshold', .5))), self.device


class LocalTexturePreviewSegmenter(LocalScratchSegmenter):
    """Semantic outlines only: disconnected pieces must never become lengths."""
    def __init__(self, model_path=None):
        super().__init__(model_path or Path(__file__).parent/'models/kaiko_texture.pt')
        if self.metadata.get('architecture') != 'kaiko_texture_mlp_v1':
            raise RuntimeError('輪郭確認モデルの形式が違います。更新ZIPを確認してください。')

    def analyze(self, frame, settings, roi=None):
        from silkworm_scratch import predict
        settings.validate()
        self.last_preview_mask = np.zeros(frame.shape[:2], np.uint8)
        view, offset = content_view(frame, roi)
        if view is None:
            return [], self.device
        probability = predict(self.model, cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
        mask = (probability >= float(self.metadata['inference_threshold'])).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= settings.min_area
        keep[0] = False
        x0, y0 = offset
        self.last_preview_mask[y0:y0+mask.shape[0], x0:x0+mask.shape[1]] = keep[labels].astype(np.uint8)*255
        result = []
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            # A semantic region can contain several touching bodies. The
            # individual-body area limit must not silently hide that region.
            if area < settings.min_area:
                continue
            local = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
            contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                result.append(Candidate(contour+np.array([x+offset[0], y+offset[1]]),
                                        None, None, 'semantic_preview_large' if area > settings.max_area else 'semantic_preview'))
        return result, self.device
