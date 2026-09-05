"""Programmatic reward: JSON-format gate over weighted coverage, action, clear-path, length, and grounding terms."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generator.prompt import parse_model_json

WEIGHTS = {"cov": 0.25, "act": 0.20, "clr": 0.20, "len": 0.15, "gnd": 0.20}
LEN_FREE = 11
LEN_SIGMA = 7.0
GND_PENALTY = 0.5
FLOOR = 0.0

GROUPS = {
    "person": "person people pedestrian passerby passersby man woman child kid crowd crowded busy throng queue line walker jogger runner shopper tourist worker student adult boy girl guy lady group folk someone couple family visitor customer diner passenger commuter",
    "cyclist": "cyclist biker motorcyclist rider",
    "bicycle": "bicycle bike cycle ebike",
    "scooter": "scooter moped vespa",
    "motorcycle": "motorcycle motorbike",
    "car": "car vehicle taxi cab sedan suv automobile jeep",
    "van": "van minivan", "bus": "bus coach minibus", "truck": "truck lorry pickup",
    "tram": "tram streetcar trolley", "train": "train subway metro",
    "traffic": "traffic roadway road street lane intersection junction roundabout crossing crosswalk highway carriageway",
    "pole": "pole post lamppost signpost streetlight streetlamp pillar column mast",
    "sign": "sign signboard signage billboard placard",
    "bollard": "bollard stanchion",
    "tree": "tree trunk branch bush shrub hedge foliage vegetation plant greenery planting",
    "stairs": "stairs stair staircase step stairway escalator",
    "curb": "curb kerb ledge dropoff",
    "water": "canal water river pond lake harbor harbour waterfront quay dock fountain sea",
    "fence": "fence railing rail barrier guardrail handrail barricade chain rope",
    "cone": "cone", "wall": "wall building facade",
    "door": "door doorway entrance gate turnstile",
    "bench": "bench seat", "furniture": "chair table stool furniture seating terrace",
    "planter": "planter flowerbed flowerpot pot",
    "trash": "trash bin garbage dumpster waste litter",
    "stall": "stall kiosk booth vendor market cart shop storefront stand display",
    "umbrella": "umbrella parasol",
    "luggage": "suitcase luggage bag backpack box crate package",
    "ladder": "ladder", "rock": "rock stone boulder debris rubble",
    "construction": "construction scaffolding scaffold excavator crane machinery worksite roadwork equipment",
    "animal": "dog pet animal cat bird pigeon",
    "stroller": "stroller pram buggy pushchair", "wheelchair": "wheelchair",
    "puddle": "puddle ice mud snow", "pothole": "pothole crack bump dip uneven slab paving cobble cobblestone tile brick unevenness",
    "manhole": "manhole grate drain grating", "ramp": "ramp slope incline",
    "hydrant": "hydrant meter", "track": "track rail",
    "obstacle": "obstacle obstruction object clutter",
}
GENERIC = {
    "vehicle": {"car", "van", "bus", "truck", "motorcycle", "tram", "train"},
    "traffic": {"car", "van", "bus", "truck", "motorcycle", "cyclist", "bicycle", "scooter", "tram"},
    "obstacle": set(GROUPS) - {"person", "traffic", "obstacle"},
}
WORD2G: dict[str, set[str]] = {}
for _g, _ws in GROUPS.items():
    for _w in _ws.split():
        WORD2G.setdefault(_w, set()).add(_g)
for _w, _gs in GENERIC.items():
    WORD2G.setdefault(_w, set()).update(_gs)

IRREG = {"people": "person", "children": "child", "men": "man", "women": "woman",
         "buses": "bus", "passersby": "passerby", "benches": "bench", "boxes": "box",
         "branches": "branch", "taxis": "taxi", "stairs": "stairs"}
STOP = set("a an the this that these those and or but if then so of in on at to for with "
           "from by your you my it its is are was were be been being as into onto up down "
           "over under near next ahead behind left right there here not no do does don't "
           "keep stay slow stop wait move step go walk turn continue proceed watch mind".split())

def lemma(w: str) -> str:
    if w in IRREG:
        return IRREG[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ses", "xes", "shes", "ches")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

def tokens(text: str) -> list[str]:
    return [lemma(w) for w in re.sub(r"[^a-z0-9' -]", " ", (text or "").lower())
            .replace("-", " ").replace("'", "").split()]

def groups_of(text: str) -> set[str]:
    out: set[str] = set()
    for w in tokens(text):
        out |= WORD2G.get(w, set())
    return out

def head_groups(phrase: str) -> set[str]:
    ws = tokens(phrase)
    if not ws:
        return set()
    return WORD2G.get(ws[-1], set()) or groups_of(phrase)

_V = r"(?:keep|stay|staying|move|moving|step|stepping|veer|veering|bear|shift|shifting|go|going|walk|walking|hug|hugging|stick|edge|swing|drift|head|heading|pass|passing|steer|steering|turn|turning|lean|favor|favour|angle|cut|skirt|sidestep)"
RE_SIDE = re.compile(rf"\b{_V}\b(?:\s+\w+){{0,3}}?\s+(left|right)\b", re.I)
RE_SIDE2 = re.compile(r"\b(?:to|toward|towards)\s+(?:the\s+|your\s+)?(left|right)\b", re.I)
RE_STRAIGHT = re.compile(
    r"\b(straight\b|continue\s+(forward|straight|walking|on|up|upward|upwards|down|downward|downwards|along|ahead|past|through)|keep\s+(going|walking|moving|advancing)|(keep|stay|remain)\w*\s+(centered|centred|center|centre|middle|on course)|proceed\s+(straight|forward|ahead|along)|carry on|walk\s+(on|forward|ahead))\b", re.I)
RE_STOP = re.compile(r"\b(stop|halt|wait|hold up|stand still|freeze|do not (move|step|proceed|go)|don'?t (move|step|proceed|go)|pause|step aside|move aside|give way|yield|let (them|him|her|it) pass)\b", re.I)
RE_SLOW = re.compile(r"\b(slow down|slow|ease up|reduce (your )?(speed|pace)|take it slow|proceed slowly|walk slowly)\b", re.I)

OPPOSITE = {frozenset({"left", "right"}), frozenset({"stop", "straight"})}

def action_of(sentence: str):

    s = sentence or ""
    m = RE_SIDE.search(s) or RE_SIDE2.search(s)
    if m:
        sides = {x.lower() for x in RE_SIDE.findall(s)} | {x.lower() for x in RE_SIDE2.findall(s)}
        if len(sides) > 1:
            return "both"
        return m.group(1).lower()
    if RE_STRAIGHT.search(s):
        return "straight"
    if RE_STOP.search(s):
        return "stop"
    if RE_SLOW.search(s):
        return "slow"
    return None

def kappa(u, v) -> float:
    if u is None or v is None:
        return 0.5
    if u == "both" or v == "both":
        return 0.0 if u == "both" else 0.5
    if u == v:
        return 1.0
    return 0.0 if frozenset({u, v}) in OPPOSITE else 0.5

RE_CLEAR = re.compile(
    r"\b(path|way|route|road|sidewalk|pavement|walkway|street|area|space)\b[^.;]{0,25}?\b"
    r"(clear|unobstructed|open|free|empty)\b"
    r"|\b(clear|unobstructed) (path|way|route|road|sidewalk|walkway|ahead)\b"
    r"|\bno (obstacles?|obstructions?|hazards?|dangers?|traffic|people)\b"
    r"|\bnothing (in your way|ahead|blocking|in the way)\b"
    r"|\ball clear\b|\bsafe to (proceed|continue|walk|go)\b"
    r"|\bit is (clear|safe)\b|\blooks clear\b|\bappears clear\b", re.I)

def claims_clear(alert: str) -> bool:
    return bool(RE_CLEAR.search(alert or ""))

class Encoder:

    NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str | None = None):
        self.device, self.tok, self.model, self.cache = device, None, None, {}

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.NAME)
        self.model = AutoModel.from_pretrained(self.NAME).eval()
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, words: list[str]):
        import torch
        todo = [w for w in dict.fromkeys(words) if w not in self.cache]
        if todo:
            if self.model is None:
                self._load()
            with torch.no_grad():
                for i in range(0, len(todo), 256):
                    ch = todo[i:i + 256]
                    enc = self.tok(ch, padding=True, truncation=True, max_length=16,
                                   return_tensors="pt").to(self.device)
                    out = self.model(**enc).last_hidden_state
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    emb = (out * m).sum(1) / m.sum(1).clamp(min=1)
                    emb = torch.nn.functional.normalize(emb, dim=-1).cpu()
                    for w, e in zip(ch, emb):
                        self.cache[w] = e
        return [self.cache[w] for w in words]

    def best_cos(self, target: str, words: list[str]) -> float:
        if not words:
            return 0.0
        import torch
        vecs = self.embed([target] + words)
        t, rest = vecs[0], torch.stack(vecs[1:])
        return float((rest @ t).max())

def r_len(alert: str) -> float:

    import math
    n = len((alert or "").split())
    return math.exp(-(max(0, n - LEN_FREE) ** 2) / (2.0 * LEN_SIGMA ** 2))

def r_act(alert: str, gold_alert: str) -> float:
    a = action_of(alert)
    return 0.5 * (1.0 if a not in (None,) else 0.0) + 0.5 * kappa(a, action_of(gold_alert))

def r_cov(alert: str, gold_hazards: list, enc: Encoder | None = None) -> float:
    if not gold_hazards:
        return 1.0
    ag = groups_of(alert)
    words = [w for w in tokens(alert) if w not in STOP and len(w) > 2]
    tot = 0.0
    for h in gold_hazards:
        name = h.get("hazard", "") if isinstance(h, dict) else str(h)
        hg = head_groups(name)
        if hg and (hg & ag):
            tot += 1.0
        elif enc is not None:
            c = enc.best_cos(name.lower(), words)
            tot += min(1.0, max(0.0, (c - 0.4) / 0.4))
    return tot / len(gold_hazards)

def r_gnd(alert: str, gold: dict, hints: str = "") -> float:

    named = groups_of(alert)
    if not named:
        return 1.0
    support = parse_hints(hints) | groups_of(gold.get("alert", ""))
    for h in gold.get("hazards") or []:
        support |= head_groups(h.get("hazard", "") if isinstance(h, dict) else str(h))
    for gname, members in GENERIC.items():
        if members & support:
            support.add(gname)
    return max(0.0, 1.0 - GND_PENALTY * len(named - support))

def parse_hints(hints: str) -> set:

    out = set()
    for part in (hints or "").split(";"):
        m = re.match(r"\s*\[\d+\]\s*([^,]+)", part)
        if m:
            out |= groups_of(m.group(1))
    return out

def r_clr(alert: str, gold_danger) -> float:
    try:
        d = int(gold_danger)
    except (TypeError, ValueError):
        d = 3
    return 0.0 if (claims_clear(alert) and d >= 3) else 1.0

def g_fmt(output: str):
    obj = parse_model_json(output or "")
    if not isinstance(obj, dict):
        return 0.0, ""
    a = obj.get("alert")
    if not isinstance(a, str) or not a.strip():
        return 0.0, ""
    return 1.0, a.strip()

def components(output: str, gold: dict, enc: Encoder | None = None,
               hints: str = "") -> dict:
    fmt, alert = g_fmt(output)
    if fmt == 0.0:
        return {"fmt": 0.0, "cov": 0.0, "act": 0.0, "clr": 0.0, "len": 0.0,
                "gnd": 0.0, "n_words": 0}
    return {"fmt": 1.0,
            "cov": r_cov(alert, gold.get("hazards") or [], enc),
            "act": r_act(alert, gold.get("alert", "")),
            "clr": r_clr(alert, gold.get("danger_level")),
            "len": r_len(alert),
            "gnd": r_gnd(alert, gold, hints),
            "n_words": len(alert.split())}

def total(comp: dict, weights: dict | None = None) -> float:
    w = weights or WEIGHTS
    if comp.get("fmt", 0.0) < 1.0:
        return FLOOR
    return sum(w[k] * comp[k] for k in w)

def load_gold(path) -> dict[str, dict]:
    out = {}
    for line in Path(path).open():
        if not line.strip():
            continue
        r = json.loads(line)
        if "label" in r:
            out[r["id"]] = r["label"]
        elif "target_json" in r:
            out[r["id"]] = json.loads(r["target_json"])
    return out
