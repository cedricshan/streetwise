"""Generator prompt template, input text builders, and JSON output parser."""
from __future__ import annotations

import json

PROMPT_TMPL = (
    "You assist a blind user. From the current first-person camera frame and "
    "the object info below, report walking hazards as one line of JSON "
    '{{"alert": ..., "hazards": [...], "scene": ..., "danger_level": ...}}.\n'
    "Objects (current frame): {detections}\n"
    "Motion (last ~1s): {motion}"
)

FRAMES_PROMPT_TMPL = (
    "You assist a blind user. From the {n} first-person camera frames below, "
    "taken 0.375 s apart in chronological order, report walking hazards as "
    'one line of JSON {{"alert": ..., "hazards": [...], "scene": ..., '
    '"danger_level": ...}}.'
)

ALERT_PROMPT_TMPL = (
    "You assist a blind user. From the current first-person camera frame and "
    "the object info below, warn about walking hazards in one short spoken "
    "sentence.\n"
    "Objects (current frame): {detections}\n"
    "Motion (last ~1s): {motion}"
)

def build_user_text(detections: str, motion: str, alert_only: bool = False) -> str:
    tmpl = ALERT_PROMPT_TMPL if alert_only else PROMPT_TMPL
    return tmpl.format(detections=detections or "none", motion=motion or "none")

def build_frames_text(n: int = 9) -> str:
    return FRAMES_PROMPT_TMPL.format(n=n)

def target_json(label: dict) -> str:

    hazards = [
        {"hazard": h.get("hazard", ""),
         "direction": h.get("direction", ""),
         "motion": h.get("motion", "")}
        for h in label.get("hazards", [])
    ]
    obj = {
        "alert": label.get("alert", ""),
        "hazards": hazards,
        "scene": label.get("scene", ""),
        "danger_level": label.get("danger_level"),
    }
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def target_alert(label: dict) -> str:
    return label.get("alert", "")

def parse_model_json(text: str) -> dict | None:

    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
