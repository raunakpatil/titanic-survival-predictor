"""Streamlit wrapper for the 3D Titanic viewer in ship_viewer/ (plain HTML + vendored three.js)."""
from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component("ship_viewer", path=str(Path(__file__).resolve().parent / "ship_viewer"))

# Where each class slept, from the ship's deck plans (approximate). 3rd-class single men were
# berthed in the bow, women and families at the stern.
BERTHS = {
    1: ("first", "B & C decks, amidships", 2),
    2: ("second", "D & E decks, aft", 4),
    (3, "male"): ("third_bow", "F & G decks, in the bow", 6),
    (3, "female"): ("third_stern", "F & G decks, at the stern", 6),
}
# Label the lifeboat group furthest from the berth so the two tags never overlap
_BOAT_LABEL = {"first": "boats_aft", "second": "boats_fwd", "third_bow": "boats_aft", "third_stern": "boats_fwd"}


def berth_for(pclass, sex):
    """(anchor, description, decks below the boat deck) for a passenger."""
    return BERTHS[(3, sex)] if pclass == 3 else BERTHS[pclass]


def ship_berth(pclass, sex, color, height=400, key=None):
    anchor, where, _ = berth_for(pclass, sex)
    labelled = _BOAT_LABEL[anchor]
    markers = [
        {"at": group, "color": "#4ade80", **({"title": "Lifeboats", "sub": "boat deck"} if group == labelled else {})}
        for group in ("boats_fwd", "boats_aft")
    ] + [
        {"at": anchor, "title": "Your berth", "sub": where, "color": color, "pulse": True},
    ]
    return _component(mode="berth", markers=markers, height=height, autorotate=True, key=key, default=None)


def ship_sinking(pitch, sink, boats, height=400, key=None):
    return _component(mode="sink", pitch=pitch, sink=sink, boats=boats, height=height,
                      autorotate=False, key=key, default=None)
