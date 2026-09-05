"""Open-vocabulary YOLO detector wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

WEIGHTS = Path("models/prep/yoloe-26x-seg.pt")

@dataclass(frozen=True)
class Det:
    label: str
    conf: float
    box: tuple[float, float, float, float]

class Detector:
    def __init__(self, classes: list[str], weights: Path = WEIGHTS,
                 imgsz: int = 960, conf: float = 0.12, device: str = "cuda"):
        import os
        weights = Path(weights).resolve()
        cwd = os.getcwd()
        os.chdir(weights.parent)
        from ultralytics import YOLOE
        self.model = YOLOE(str(weights))
        self.model.set_classes(classes, self.model.get_text_pe(classes))
        os.chdir(cwd)
        self.model.to(device)
        self.imgsz = imgsz
        self.conf = conf

    def detect_clip(self, frame_paths: list) -> list[list[Det]]:

        results = self.model.predict(
            [str(p) for p in frame_paths], imgsz=self.imgsz, conf=self.conf,
            iou=0.45, verbose=False)
        out: list[list[Det]] = []
        for r in results:
            dets = []
            names = r.names
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                dets.append(Det(label=names[int(b.cls)], conf=float(b.conf),
                                box=(x1, y1, x2, y2)))
            out.append(dets)
        return out
