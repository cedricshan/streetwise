"""Greedy IoU association and per-track velocity fits in metric camera space."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .detect import Det
from .vocab import SURFACE, THREAT

DT = 0.375
STATIC_SPEED = 0.5
FAST_SPEED = 2.0
CORRIDOR_HALF = 0.9

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua

@dataclass
class Track:
    tid: int
    label: str
    obs: list = field(default_factory=list)

    @property
    def last_box(self):
        return self.obs[-1][1]

    @property
    def last_frame(self) -> int:
        return self.obs[-1][0]

    def xyz_series(self):
        return [(k * DT, p) for k, _, _, p in self.obs if p is not None]

def associate(per_frame_dets: list[list[Det]],
              per_frame_pos: list[list[tuple | None]],
              cmcs: list, use_cmc: bool = True) -> list[Track]:

    from perception.cmc import warp_box

    tracks: list[Track] = []
    next_id = 0
    for k, (dets, poss) in enumerate(zip(per_frame_dets, per_frame_pos)):
        live = [t for t in tracks if k - t.last_frame <= 2]
        warped = {}
        for t in live:
            box = t.last_box
            if use_cmc:
                for j in range(t.last_frame, k):
                    box = warp_box(cmcs[j], box)
            warped[t.tid] = box
        cand = []
        for di, d in enumerate(dets):
            for t in live:
                iou = _iou(warped[t.tid], d.box)
                if iou >= 0.2:
                    bonus = 0.2 if t.label == d.label else 0.0
                    cand.append((iou + bonus, di, t.tid))
        cand.sort(reverse=True)
        used_d, used_t = set(), set()
        tmap = {t.tid: t for t in live}
        for score, di, tid in cand:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di); used_t.add(tid)
            tmap[tid].obs.append((k, dets[di].box, dets[di].conf,
                                  per_frame_pos[k][di]))
        for di, d in enumerate(dets):
            if di not in used_d:
                tracks.append(Track(next_id, d.label,
                                    [(k, d.box, d.conf, per_frame_pos[k][di])]))
                next_id += 1
    return tracks

def _fit_velocity(series, n_last: int = 6):

    series = series[-n_last:]
    t = np.array([s[0] for s in series])
    xs = np.array([s[1][0] for s in series])
    zs = np.array([s[1][2] for s in series])
    if len(series) < 3 or t[-1] - t[0] < 1e-6:
        return float(xs[-1]), float(zs[-1]), 0.0, 0.0, False
    A = np.stack([t - t[-1], np.ones_like(t)], axis=1)
    (vx, x0), _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
    (vz, z0), _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
    return float(x0), float(z0), float(vx), float(vz), True

def clock_of(x: float, z: float) -> str:
    deg = math.degrees(math.atan2(x, max(z, 0.05)))
    for lim, name in ((-75, "9 o'clock"), (-45, "10 o'clock"),
                      (-15, "11 o'clock"), (15, "12 o'clock"),
                      (45, "1 o'clock"), (75, "2 o'clock")):
        if deg < lim:
            return name
    return "3 o'clock"

def analyze(tracks: list[Track], ego_forward_hint: bool,
            final_frame_size: tuple[int, int],
            ego_off: bool = False,
            now_frame: int = 8) -> tuple[list[dict], float]:

    W, H = final_frame_size
    fits = {}
    for t in tracks:
        s = t.xyz_series()
        if s:
            fits[t.tid] = _fit_velocity(s)

    vels = [(f[2], f[3], f[1]) for f in fits.values() if f[4] and f[1] > 0.5]
    a_yaw = (float(np.median([vx / max(z, 0.5) for vx, _, z in vels]))
             if len(vels) >= 3 else 0.0)
    if ego_off:
        a_yaw = ego_speed = 0.0
    elif not ego_forward_hint:
        ego_speed = 0.0
    elif len(vels) >= 3:
        ego_speed = float(np.clip(np.median([-vz for _, vz, _ in vels]),
                                  0.0, 2.5))
    else:
        ego_speed = 1.2

    out = []
    for t in tracks:
        if t.last_frame != now_frame or t.tid not in fits:
            continue
        x0, z0, vx, vz, has_v = fits[t.tid]
        if z0 < 0.2:
            continue
        world_vx = vx - a_yaw * z0 if has_v else 0.0
        world_vz = vz + ego_speed if has_v else 0.0
        speed = math.hypot(world_vx, world_vz) if has_v else 0.0
        is_static = (t.label in SURFACE) or (not has_v) or (speed < STATIC_SPEED)

        ttc = miss = None
        if has_v and vz < -0.15:
            ttc = z0 / -vz
            miss = x0 + vx * ttc
        elif is_static and ego_speed > 0.3:
            ttc = z0 / ego_speed
            miss = x0
        if ttc is not None and ttc <= 0:
            ttc = miss = None
        ttc_eff = min(ttc, 30.0) if ttc is not None else 30.0
        miss_abs = abs(miss) if miss is not None else abs(x0)

        if is_static:
            motion = "static"
        elif abs(world_vz) >= abs(world_vx):
            motion = "approaching" if world_vz < 0 else "receding"
            if motion == "approaching" and speed > FAST_SPEED:
                motion = "approaching fast"
        else:
            motion = ("crossing left-to-right" if world_vx > 0
                      else "crossing right-to-left")

        corridor = (1.5 if miss_abs < CORRIDOR_HALF
                    else 1.0 if miss_abs < 2 * CORRIDOR_HALF else 0.35)
        urgency = THREAT.get(t.label, 1.0) * corridor / max(ttc_eff, 0.7)

        box = t.last_box
        out.append({
            "tid": t.tid,
            "label": t.label, "n_obs": len(t.obs),
            "conf": round(max(c for _, _, c, _ in t.obs), 3),
            "box": [round(v, 1) for v in box],
            "cx": round((box[0] + box[2]) / 2 / W, 2),
            "cy": round((box[1] + box[3]) / 2 / H, 2),
            "clock": clock_of(x0, z0),
            "dist": round(math.hypot(x0, z0), 1),
            "speed": round(speed, 2), "motion": motion,
            "ttc": round(ttc, 1) if ttc is not None else None,
            "miss": round(miss, 2) if miss is not None else None,
            "urgency": round(urgency, 4),

            "x": round(x0, 3), "z": round(z0, 3),
            "vx": round(world_vx, 3), "vz": round(world_vz, 3),
            "hfrac": round((box[3] - box[1]) / H, 4),
            "wfrac": round((box[2] - box[0]) / W, 4),
        })
    out.sort(key=lambda d: -d["urgency"])
    for i, d in enumerate(out):
        d["mark"] = i + 1
    return out, ego_speed
