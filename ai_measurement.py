"""Optional local MobileSAM proposals. Geometry checks do not confirm silkworm identity."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from measurement import Candidate, centerline, polyline_length

MODEL_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
MODEL_SHA256 = "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
MODEL_URL = f"https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/{MODEL_COMMIT}/weights/mobile_sam.pt"


def content_view(frame, roi=None):
    """Trim black letterboxing while retaining source-image coordinates."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi or (0, 0, w, h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("測定範囲が空です。")
    view = frame[y0:y1, x0:x1]
    dark = view.max(axis=2) <= 8
    rows = np.flatnonzero(np.any(~dark, axis=1))
    cols = np.flatnonzero(np.any(~dark, axis=0))
    if not len(rows) or not len(cols):
        return None, (x0, y0)
    left, right = int(cols[0]), int(cols[-1]) + 1
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    return view[top:bottom, left:right], (x0 + left, y0 + top)


def candidates_from_masks(view, regions, settings, offset=(0, 0)):
    """Use independent masks instead of merging every white pixel into one region."""
    settings.validate()
    hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
    proposals = []
    h, w = view.shape[:2]
    for region in regions:
        if region.get("predicted_iou", 0) < .88 or region.get("stability_score", 0) < .92:
            continue
        mask = np.asarray(region["segmentation"], dtype=np.uint8)
        if mask.shape != (h, w):
            raise ValueError("AIの領域と入力画像のサイズが一致しません。")
        area = int(mask.sum())
        if not settings.min_area <= area <= settings.max_area:
            continue
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        x, y, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
        part = mask[y:y1, x:x1]
        selected = hsv[y:y1, x:x1][part > 0]
        median_s, median_v = np.median(selected[:, 1:], axis=0)
        if median_s > settings.saturation:
            continue
        if settings.appearance == "light" and median_v < settings.brightness:
            continue
        if settings.appearance == "dark" and median_v > settings.brightness:
            continue
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) + np.array([x+offset[0], y+offset[1]])
        path, length, reason = None, None, "edge"
        if x > 2 and y > 2 and x1 < w-2 and y1 < h-2:
            path, reason = centerline(np.pad(part, 2))
            if path is not None:
                length = polyline_length(path)
                if length*length/area < settings.min_aspect:
                    path, length, reason = None, None, "shape"
                else:
                    path += np.array([x+offset[0]-2, y+offset[1]-2])
        proposals.append({"candidate": Candidate(contour, path, length, reason),
                          "mask": part.astype(bool), "area": area, "box": (x, y, x1, y1)})

    # A fragment inside a complete accepted mask must not be counted twice.
    accepted = []
    for p in sorted(proposals, key=lambda v: v["area"], reverse=True):
        c = p["candidate"]
        if c.length_px is None:
            continue
        for q in accepted:
            ax, ay, ax1, ay1 = p["box"]
            bx, by, bx1, by1 = q["box"]
            left, top, right, bottom = max(ax, bx), max(ay, by), min(ax1, bx1), min(ay1, by1)
            if left >= right or top >= bottom:
                continue
            intersection = np.count_nonzero(
                p["mask"][top-ay:bottom-ay, left-ax:right-ax] &
                q["mask"][top-by:bottom-by, left-bx:right-bx])
            if intersection / min(p["area"], q["area"]) >= .90:
                c.path, c.length_px, c.reason = None, None, "duplicate_or_part"
                break
        if c.length_px is not None:
            accepted.append(p)

    # Reject much thinner proposals when enough similarly sized bodies provide a scale.
    # This is a heuristic for a single cohort, not a species classifier.
    if len(accepted) >= 5:
        width = float(np.median([p["area"]/p["candidate"].length_px for p in accepted]))
        for p in accepted:
            c = p["candidate"]
            if p["area"]/c.length_px < .5*width:
                c.path, c.length_px, c.reason = None, None, "relative_scale"
    return [p["candidate"] for p in proposals]


class LocalSegmenter:
    def __init__(self, checkpoint=None):
        checkpoint = Path(checkpoint or Path(__file__).resolve().parent / "models/mobile_sam.pt")
        if not checkpoint.is_file():
            raise RuntimeError("AI追加セットアップが必要です。アプリを閉じて setup_ai_windows.bat を実行してください。")
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError("AIモデルの内容が一致しません。setup_ai_windows.bat で取得し直してください。")
        try:
            import torch
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        except (ImportError, OSError) as exc:
            raise RuntimeError("AIライブラリを読み込めません。setup_ai_windows.bat を実行し、アプリを再起動してください。") from exc
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(3)
        model = sam_model_registry["vit_t"]()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.generator = SamAutomaticMaskGenerator(
            model, points_per_side=16, points_per_batch=8,
            pred_iou_thresh=.88, stability_score_thresh=.92, min_mask_region_area=250)

    def analyze(self, frame, settings, roi=None):
        view, offset = content_view(frame, roi)
        if view is None:
            return [], self.device
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        regions = self.generator.generate(rgb)
        return candidates_from_masks(view, regions, settings, offset), self.device


@dataclass
class AIJob:
    frame: np.ndarray
    position: float | None
    settings: object
    roi: tuple | None
    token: object


class AIWorker:
    """One job at a time. Never run Torch or touch Tk widgets on the UI thread."""
    def __init__(self, factory=LocalSegmenter):
        self.factory = factory
        self.engine = None
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.busy = False
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, job):
        if self.busy or self.stopped.is_set():
            return False
        self.busy = True
        self.jobs.put_nowait(job)
        return True

    def receive(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def run(self):
        while not self.stopped.is_set():
            try:
                job = self.jobs.get(timeout=.1)
            except queue.Empty:
                continue
            start = time.monotonic()
            try:
                if self.engine is None:
                    self.engine = self.factory()
                candidates, device = self.engine.analyze(job.frame, job.settings, job.roi)
                result = {"job": job, "candidates": candidates, "device": device,
                          "seconds": time.monotonic()-start, "error": None}
            except Exception as exc:
                result = {"job": job, "error": str(exc) or type(exc).__name__}
            if not self.stopped.is_set():
                self.results.put_nowait(result)

    def close(self):
        self.stopped.set()
