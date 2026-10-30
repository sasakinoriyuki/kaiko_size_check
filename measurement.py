"""Conservative, untrained image-processing baseline. All geometry is in source pixels."""
from dataclasses import dataclass, asdict
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class Settings:
    brightness: int = 145
    saturation: int = 90
    min_area: int = 100
    max_area: int = 100000
    min_aspect: float = 3.0
    appearance: str = "light"
    ai_search: str = "standard"

    def validate(self):
        if self.ai_search not in {"standard", "fine"}:
            raise ValueError("AI探索は通常／詳細から選んでください。")
        if self.appearance not in {"light", "dark"}:
            raise ValueError("検出対象は明るい蚕／暗い蚕から選んでください。")
        if not (0 <= self.brightness <= 255 and 0 <= self.saturation <= 255):
            raise ValueError("明るさ・彩度は0〜255です。")
        if not (10 <= self.min_area < self.max_area):
            raise ValueError("面積は 10 ≤ 最小 < 最大 にしてください。")
        if not 2 <= self.min_aspect <= 20:
            raise ValueError("細長さは2〜20です。")

    def dictionary(self):
        return asdict(self)


@dataclass
class Candidate:
    contour: np.ndarray
    path: np.ndarray | None
    length_px: float | None
    reason: str


def polyline_length(points):
    a = np.asarray(points, dtype=float)
    if len(a) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())


def calibrate(points, known_mm):
    if not math.isfinite(known_mm) or known_mm <= 0:
        raise ValueError("基準の長さは正の数で入力してください。")
    if len(points) != 2:
        raise ValueError("基準線の両端を選んでください。")
    pixels = polyline_length(points)
    if pixels < 10:
        raise ValueError("基準線が短すぎます。10px以上離して選んでください。")
    pixels_per_mm = pixels / known_mm
    if not math.isfinite(pixels_per_mm) or pixels_per_mm <= 0:
        raise ValueError("基準の長さを確認してください。")
    return pixels_per_mm


def channel_roi(frame_shape, layout="wide"):
    """Explicit preset for the supplied OBS layout; not for a direct-camera input."""
    height, width = frame_shape[:2]
    fractions = {"wide": 0.77, "two_thirds": 2 / 3}
    if layout not in fractions:
        raise ValueError("測定範囲の配置が不明です。")
    return (0, 0, max(1, int(width * fractions[layout])), height)


def _graph(skeleton):
    nodes = set(map(tuple, np.argwhere(skeleton)))
    graph = {}
    for y, x in nodes:
        neighbors = []
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            p = (y + dy, x + dx)
            if p not in nodes:
                continue
            # Do not create triangle-shaped branches around an orthogonal bend.
            if dy and dx and ((y + dy, x) in nodes or (y, x + dx) in nodes):
                continue
            neighbors.append(p)
        graph[(y, x)] = neighbors
    return graph


def _extend(point, direction, mask):
    norm = np.linalg.norm(direction)
    if norm < 0.1:
        return point
    direction = direction / norm
    end = point.copy()
    for step in np.arange(0.5, max(mask.shape) + 1, 0.5):
        p = point + direction * step
        x, y = np.rint(p).astype(int)
        if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1] or not mask[y, x]:
            break
        end = p
    return end


def centerline(mask, prune_short_spurs=False):
    """Accept only a single open, nonbranching skeleton; never infer a hidden body."""
    skeleton = skeletonize(mask > 0)
    if not prune_short_spurs:
        # Reject clearly branched regions in arrays before constructing a Python
        # graph. This uses the same diagonal-triangle rule as _graph.
        p = np.pad(skeleton, 1)
        up, down = p[:-2, 1:-1], p[2:, 1:-1]
        left, right = p[1:-1, :-2], p[1:-1, 2:]
        degree = up.astype(np.uint8)+down+left+right
        for diagonal, vertical, horizontal in ((p[:-2, :-2], up, left),
                                               (p[:-2, 2:], up, right),
                                               (p[2:, :-2], down, left),
                                               (p[2:, 2:], down, right)):
            degree += diagonal & ~vertical & ~horizontal
        if np.count_nonzero(skeleton & (degree == 1)) != 2 or np.any(skeleton & (degree > 2)):
            return None, "branch_or_loop"
    graph = _graph(skeleton)
    if prune_short_spurs:
        # Only for a user-reviewed mask: trim short skeleton twigs caused by its
        # ragged outline. Keep long branches and loops rejected, without joining gaps.
        radii = distance_transform_edt(mask > 0)
        for _ in range(8):
            remove = set()
            for end in [p for p, n in graph.items() if len(n) == 1]:
                route, previous, current, length = [end], None, end, 0.0
                while True:
                    choices = [p for p in graph[current] if p != previous]
                    if len(choices) != 1:
                        break
                    next_node = choices[0]
                    length += float(np.linalg.norm(np.subtract(next_node, current)))
                    previous, current = current, next_node
                    if len(graph[current]) != 2:
                        break
                    route.append(current)
                if len(graph[current]) > 2 and length <= radii[current]:
                    remove.update(route)
            if not remove:
                break
            graph = {p: [n for n in neighbors if n not in remove]
                     for p, neighbors in graph.items() if p not in remove}
    ends = [p for p, n in graph.items() if len(n) == 1]
    if len(ends) != 2 or any(len(n) > 2 for n in graph.values()):
        return None, "branch_or_loop"
    path = [ends[0]]
    previous = None
    while path[-1] != ends[1]:
        choices = [p for p in graph[path[-1]] if p != previous]
        if len(choices) != 1 or len(path) > len(graph):
            return None, "disconnected"
        previous = path[-1]
        path.append(choices[0])
    if len(path) != len(graph) or len(path) < 5:
        return None, "short_or_disconnected"
    xy = np.array([(x, y) for y, x in path], dtype=float)
    smoothed = gaussian_filter1d(xy, sigma=1.2, axis=0, mode="nearest")
    smoothed[0], smoothed[-1] = xy[0], xy[-1]
    n = min(5, len(xy) - 1)
    start = _extend(xy[0], xy[0] - xy[n], mask)
    end = _extend(xy[-1], xy[-1] - xy[-1 - n], mask)
    return np.vstack([start, smoothed, end]), "ok"


def analyze(frame, settings=None, roi=None):
    settings = settings or Settings()
    settings.validate()
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = roi or (0, 0, width, height)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("測定範囲が空です。")
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    luminance_selection = (value >= settings.brightness if settings.appearance == "light"
                           else value <= settings.brightness)
    mask = ((hsv[:, :, 1] <= settings.saturation) & luminance_selection).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < settings.min_area:
            continue
        part = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) + np.array([x+x0, y+y0])
        if x == 0 or y == 0 or x+w == mask.shape[1] or y+h == mask.shape[0]:
            result.append(Candidate(contour, None, None, "edge"))
            continue
        if area > settings.max_area:
            result.append(Candidate(contour, None, None, "area"))
            continue
        # Pad to keep skeleton behavior independent of component bounding boxes.
        padded = np.pad(part, 2)
        path, reason = centerline(padded)
        length = polyline_length(path) if path is not None else None
        if length is not None:
            average_width = area / max(length, 1)
            if length / max(average_width, 1) < settings.min_aspect:
                path, length, reason = None, None, "shape"
            else:
                path += np.array([x+x0-2, y+y0-2])
        result.append(Candidate(contour, path, length, reason))
    return result, mask


def summarize(candidates, pixels_per_mm=None):
    values = [c.length_px for c in candidates if c.length_px is not None]
    if pixels_per_mm is not None:
        values = [n / pixels_per_mm for n in values]
    return {
        "count": len(values), "excluded_regions": len(candidates) - len(values),
        "unit": "mm" if pixels_per_mm is not None else "px",
        "mean": float(np.mean(values)) if values else None,
        "min": min(values) if values else None, "max": max(values) if values else None,
        "lengths": values,
    }


def demo_frame():
    """Synthetic input for installation checks; not accuracy evidence on real silkworms."""
    frame = np.full((540, 960, 3), (42, 72, 40), np.uint8)
    for points, thickness in [([(100, 130), (300, 130)], 20),
                               ([(420, 100), (460, 130), (490, 190), (550, 220)], 18),
                               ([(100, 330), (320, 450)], 20),
                               ([(120, 450), (310, 320)], 20)]:
        pts = np.array(points, np.int32)
        cv2.polylines(frame, [pts], False, (225, 230, 230), thickness, cv2.LINE_AA)
        for p in (points[0], points[-1]):
            cv2.circle(frame, p, thickness // 2, (225, 230, 230), -1)
    cv2.line(frame, (700, 450), (800, 450), (0, 220, 240), 3)
    for x in (700, 800):
        cv2.line(frame, (x, 440), (x, 460), (0, 220, 240), 3)
    cv2.putText(frame, "DEMO ONLY  |  yellow line = 100 px", (550, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 240), 1)
    return frame
