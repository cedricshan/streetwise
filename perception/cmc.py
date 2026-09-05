"""Camera-motion compensation between consecutive frames: sparse LK flow with ORB and median-displacement fallbacks."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

Box = tuple[float, float, float, float]

_MAX_DIM = 480
_GFTT = dict(maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=7)
_LK = dict(winSize=(21, 21), maxLevel=5)
_MIN_POINTS = 12
_RANSAC_REPROJ_PX = 3.0

@dataclass(frozen=True)
class CmcResult:

    matrix: np.ndarray
    n_inliers: int
    quality: float
    method: str

    @property
    def linear(self) -> np.ndarray:
        return self.matrix[:, :2]

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:, 2]

    @property
    def scale(self) -> float:

        det = float(np.linalg.det(self.linear))
        return math.sqrt(det) if det > 0 else 1.0

    @property
    def rotation_rad(self) -> float:
        return math.atan2(float(self.matrix[1, 0]), float(self.matrix[0, 0]))

def identity_cmc() -> CmcResult:
    return CmcResult(np.hstack([np.eye(2), np.zeros((2, 1))]), 0, 0.0, "identity")

def warp_point(cmc: CmcResult, x: float, y: float) -> tuple[float, float]:
    p = cmc.linear @ np.array([x, y]) + cmc.translation
    return float(p[0]), float(p[1])

def warp_box(cmc: CmcResult, box: Box) -> Box:
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float64)
    warped = corners @ cmc.linear.T + cmc.translation
    return (float(warped[:, 0].min()), float(warped[:, 1].min()),
            float(warped[:, 0].max()), float(warped[:, 1].max()))

def _mask_excluding_boxes(shape: tuple[int, int], boxes: list[Box], scale: float) -> np.ndarray:
    mask = np.full(shape, 255, dtype=np.uint8)
    h, w = shape
    for x1, y1, x2, y2 in boxes:
        xa, ya = max(0, int(x1 * scale) - 2), max(0, int(y1 * scale) - 2)
        xb, yb = min(w, int(x2 * scale) + 2), min(h, int(y2 * scale) + 2)
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 0
    return mask

def _fit_similarity(prev_pts: np.ndarray, curr_pts: np.ndarray, scale: float,
                    method: str) -> CmcResult | None:
    import cv2

    if len(prev_pts) < _MIN_POINTS:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        prev_pts, curr_pts, method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_PX,
    )
    if matrix is None or inliers is None:
        return None
    n_in = int(inliers.sum())
    if n_in < _MIN_POINTS:
        return None

    full = matrix.astype(np.float64).copy()
    full[:, 2] /= scale
    return CmcResult(full, n_in, n_in / len(prev_pts), method)

def estimate_cmc(prev_gray: np.ndarray, curr_gray: np.ndarray,
                 exclude_boxes: list[Box] | None = None) -> CmcResult:

    import cv2

    scale = min(1.0, _MAX_DIM / max(prev_gray.shape))
    if scale < 1.0:
        size = (int(prev_gray.shape[1] * scale), int(prev_gray.shape[0] * scale))
        prev_s = cv2.resize(prev_gray, size, interpolation=cv2.INTER_AREA)
        curr_s = cv2.resize(curr_gray, size, interpolation=cv2.INTER_AREA)
    else:
        prev_s, curr_s = prev_gray, curr_gray
    mask = _mask_excluding_boxes(prev_s.shape, exclude_boxes or [], scale)

    p0 = cv2.goodFeaturesToTrack(prev_s, mask=mask, **_GFTT)
    if p0 is not None and len(p0) >= _MIN_POINTS:
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_s, curr_s, p0, None, **_LK)
        if p1 is not None:
            ok = status.reshape(-1) == 1
            result = _fit_similarity(p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok],
                                     scale, "lk")
            if result is not None:
                return result

    orb = cv2.ORB_create(nfeatures=500)
    kp0, des0 = orb.detectAndCompute(prev_s, mask)
    kp1, des1 = orb.detectAndCompute(curr_s, None)
    if des0 is not None and des1 is not None and len(kp0) >= _MIN_POINTS:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(des0, des1), key=lambda m: m.distance)[:200]
        if len(matches) >= _MIN_POINTS:
            pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
            pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
            result = _fit_similarity(pts0, pts1, scale, "orb")
            if result is not None:
                return result

            med = np.median(pts1 - pts0, axis=0) / scale
            matrix = np.hstack([np.eye(2), med.reshape(2, 1)])
            return CmcResult(matrix, len(matches), 0.1, "median")

    return identity_cmc()

@dataclass(frozen=True)
class EgoState:
    forward: bool
    zoom_rate: float
    turn: str | None
    turn_rate: float
    scanning: bool
    quality: float

    def describe(self) -> str:
        if self.scanning:
            return "looking around"
        parts = ["walking forward"] if self.forward else ["standing"]
        if self.turn:
            parts.append(f"turning {self.turn}")
        return ", ".join(parts)

class EgoMotionEstimator:

    def __init__(self, frame_width: int, window_s: float = 1.2,
                 forward_zoom_rate: float = 0.02, turn_rate_thresh: float = 0.05,
                 scan_rate_thresh: float = 0.35) -> None:
        self._w = float(frame_width)
        self._window_s = window_s
        self._forward_zoom_rate = forward_zoom_rate
        self._turn_rate_thresh = turn_rate_thresh
        self._scan_rate_thresh = scan_rate_thresh
        self._pairs: deque[tuple[float, float, float, float]] = deque(maxlen=64)

    def add(self, cmc: CmcResult, t: float, dt: float) -> None:
        if dt <= 0:
            return

        cx = self._w / 2.0
        tx_center = float(cmc.linear[0, 0] * cx + cmc.linear[0, 1] * 0.0
                          + cmc.translation[0] - cx)
        self._pairs.append((t, (cmc.scale - 1.0) / dt,
                            (tx_center / self._w) / dt, cmc.quality))

    def state(self, now: float) -> EgoState:
        recent = [p for p in self._pairs if now - p[0] <= self._window_s]
        if not recent:
            return EgoState(False, 0.0, None, 0.0, False, 0.0)
        zoom = float(np.median([p[1] for p in recent]))
        turn = float(np.median([p[2] for p in recent]))
        quality = float(min(p[3] for p in recent))
        scanning = abs(turn) > self._scan_rate_thresh
        direction = None
        if abs(turn) > self._turn_rate_thresh:
            direction = "left" if turn > 0 else "right"
        return EgoState(
            forward=zoom > self._forward_zoom_rate,
            zoom_rate=zoom, turn=direction, turn_rate=turn,
            scanning=scanning, quality=quality,
        )
