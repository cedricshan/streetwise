"""Formats ranked tracks and ego state into the object and motion text lines and draws the numbered marks."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

MAX_OBJECTS = 8
_FONT_CANDIDATES = (
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

def objects_text(tracks: list[dict]) -> str:
    parts = []
    for d in tracks[:MAX_OBJECTS]:
        parts.append(f"[{d['mark']}] {d['label']}, {d['clock']}, "
                     f"{d['dist']}m ({d['cx']:.2f},{d['cy']:.2f})")
    return "; ".join(parts) if parts else "none"

def motion_text(tracks: list[dict], ego_desc: str | None,
                ego_speed: float) -> str:

    if ego_desc and "walking forward" in ego_desc and ego_speed > 0.05:
        ego_desc = ego_desc.replace(
            "walking forward", f"walking forward ~{ego_speed:.1f} m/s")
    parts = []
    for d in tracks[:MAX_OBJECTS]:
        if d["motion"] == "static":
            seg = f"[{d['mark']}] static"
        else:
            speed = min(d["speed"], 15.0)
            seg = f"[{d['mark']}] {d['motion']} {speed:.1f} m/s"
        if d["ttc"] is not None and d["ttc"] <= 10:
            ttc = f"<1s" if d["ttc"] < 1 else f"~{d['ttc']:.0f}s"
            seg += f", TTC {ttc}"
        parts.append(seg)
    obj = "; ".join(parts) if parts else "no tracked objects"
    return f"Ego: {ego_desc}. {obj}" if ego_desc else obj

def _font(size: int):
    for p in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()

def draw_marks(img: Image.Image, tracks: list[dict]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    lw = max(2, W // 320)
    font = _font(max(16, H // 30))
    for d in tracks[:MAX_OBJECTS]:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(50, 255, 50), width=lw)
        tag = str(d["mark"])
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = lw * 2
        bx1, by1 = x1, max(0, y1 - th - 3 * pad)
        draw.rectangle([bx1, by1, bx1 + tw + 2 * pad, by1 + th + 3 * pad],
                       fill=(50, 255, 50))
        draw.text((bx1 + pad, by1 + pad), tag, fill=(0, 0, 0), font=font)
    return out
