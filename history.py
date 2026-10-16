"""The night of 14–15 April 1912, minute by minute.

Times are approximate: the British and US inquiries and later research disagree by several
minutes for many lifeboats. Minutes are counted from the collision at 23:40.
"""
import numpy as np

COLLISION = (23, 40)
END = 160  # 02:20, the ship is gone

# (boat, minutes after collision, seats)
LIFEBOATS = [
    ("7", 65, 65), ("5", 75, 65), ("3", 80, 65), ("8", 80, 65), ("1", 85, 40), ("6", 90, 65),
    ("16", 100, 65), ("14", 110, 65), ("12", 110, 65), ("9", 110, 65), ("11", 115, 65),
    ("C", 120, 47), ("13", 120, 65), ("15", 121, 65), ("2", 125, 40), ("10", 130, 65),
    ("4", 130, 65), ("D", 145, 47), ("A", 155, 47), ("B", 155, 47),
]
TOTAL_SEATS = sum(s for _, _, s in LIFEBOATS)  # 1,178
ABOARD = 2224
SURVIVORS = 710

EVENTS = [
    (0, "Titanic strikes an iceberg on her starboard bow"),
    (25, "Captain Smith orders the lifeboats uncovered"),
    (35, "The first distress call goes out by wireless"),
    (65, "Lifeboat 7 is lowered, the first to leave. Distress rockets begin"),
    (85, "Lifeboat 1 leaves with 12 people in 40 seats"),
    (145, "Collapsible D, the last boat lowered, leaves. About 1,500 people remain aboard"),
    (155, "Collapsibles A and B float off as the water reaches the boat deck"),
    (157, "The lights fail and the ship breaks apart"),
    (160, "Titanic sinks. About 710 people survive, most of them in the lifeboats"),
]

# Illustrative pitch (degrees, bow down) and sinking depth (model units) over time
_CURVE_T = [0, 30, 60, 90, 120, 135, 145, 150, 155, 158, 160]
_PITCH = [0, 0.5, 1.5, 3, 5, 8, 11, 14, 22, 35, 60]
_SINK = [0, 1, 2, 4, 7, 10, 13, 17, 24, 40, 160]


def clock(minute):
    total = COLLISION[0] * 60 + COLLISION[1] + minute
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


def state_at(minute):
    launched = [b for b in LIFEBOATS if b[1] <= minute]
    event = [e for e in EVENTS if e[0] <= minute][-1]
    return {
        "pitch": float(np.interp(minute, _CURVE_T, _PITCH)),
        "sink": float(np.interp(minute, _CURVE_T, _SINK)),
        "boats": len(launched),
        "seats": sum(s for *_, s in launched),
        "event": event,
    }
