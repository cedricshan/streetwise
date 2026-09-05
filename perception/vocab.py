"""Street-scene detection vocabulary and per-class threat weights."""
CLASSES = [

    "person", "stroller", "wheelchair", "dog",
    "bicycle", "scooter", "motorcycle",

    "car", "van", "bus", "truck", "tram", "train",

    "pole", "bollard", "tree trunk", "traffic sign", "street lamp",
    "parking meter", "fire hydrant", "pillar", "utility box", "mailbox",

    "stairs", "curb", "ramp", "speed bump", "manhole", "pothole", "puddle",
    "railway track", "grate",

    "fence", "railing", "barrier", "construction barrier", "traffic cone",
    "wall", "gate", "glass door", "door", "canal", "water",

    "bench", "chair", "table", "planter", "trash can", "market stall",
    "kiosk", "signboard", "umbrella", "suitcase", "box", "crate",
    "shopping cart", "ladder", "rock", "construction equipment",
]

THREAT = {}
for c in ("car", "van", "bus", "truck", "tram", "train", "motorcycle"):
    THREAT[c] = 3.0
for c in ("bicycle", "scooter"):
    THREAT[c] = 2.2
for c in ("person", "stroller", "wheelchair", "dog"):
    THREAT[c] = 1.5
for c in ("stairs", "curb", "canal", "water", "pothole", "railway track"):
    THREAT[c] = 1.8

SURFACE = {"stairs", "curb", "ramp", "speed bump", "manhole", "pothole",
           "puddle", "railway track", "grate", "canal", "water", "wall"}
