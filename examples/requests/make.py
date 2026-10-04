"""Writes the example requests E1 to E7 (plan 4.11) as JSON files next to this script.

Run: uv run python examples/requests/make.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def w(text, t0, t1):
    return {"text": text, "t0": t0, "t1": t1}


def words(spec):
    return [w(t, a, b) for t, a, b in spec]


def hold(t0, t1, x, y, frame, source, spread=(0.8, 0.9)):
    return {"t0": t0, "t1": t1, "mean": [x, y], "spread": list(spread), "frame": frame,
            "source": source}  # fmt: skip


BLOCKS = [
    {"id": "red", "label": "red block", "box_mm": [125, 225, 50, 50], "attrs": {"color": "red"},
     "height_mm": 50, "t": 1},
    {"id": "yellow", "label": "yellow block", "box_mm": [215, 225, 50, 50],
     "attrs": {"color": "yellow"}, "height_mm": 50, "t": 2},
    {"id": "green", "label": "green block", "box_mm": [300, 100, 50, 50],
     "attrs": {"color": "green"}, "height_mm": 50, "t": 3},
    {"id": "blue", "label": "blue block", "box_mm": [420, 260, 50, 50], "attrs": {"color": "blue"},
     "height_mm": 50, "t": 4},
]  # fmt: skip

E4_PLACE = {
    "version": "1.0", "frame": "table", "kind": "box", "mean_mm": [480.0, 90.0],
    "box_mm": [455.0, 65.0, 50.0, 50.0], "cov_mm2": [[25.0, 0.0], [0.0, 25.0]],
    "source": "gesture", "evidence": {"rule": "k_in_order", "box_from": "object"},
}  # fmt: skip

REQUESTS = {
    "E1-plotter-cat-here-this-big": {
        "profile": "plotter-a5",
        "request": {
            "version": "1.0",
            "words": words([("Draw", 10.0, 10.3), ("a", 10.3, 10.4), ("cat", 10.4, 10.8),
                            ("here", 10.8, 11.1), ("this", 11.6, 11.8), ("big", 11.8, 12.1)]),
            "holds": [hold(10.2, 10.95, 112.4, 80.2, "paper", "finger")],
            "spans": [{"t0": 11.4, "t1": 12.0, "span_mm": 52.0, "kind": "pinch",
                       "center": [118.0, 84.0]}],
            "things": [],
            "calls": [{"name": "place_drawing", "args": {
                "object": "cat", "place": {"deixis": "here", "word_index": 3},
                "size": {"word_index": 4}}}],
        },
    },
    "E2-vacuum-clean-here": {
        "profile": "vacuum-floor",
        "request": {
            "version": "1.0",
            "words": words([("clean", 1.0, 1.3), ("here", 1.3, 1.6)]),
            "samples": [{"t": round(k / 30, 4), "point": [1520.0, 980.0]} for k in range(60)],
            "calls": [{"name": "clean", "args": {"place": {"deixis": "here", "word_index": 1}}}],
        },
    },
    "E3-vacuum-clean-from-here-to-here": {
        "profile": "vacuum-floor",
        "request": {
            "version": "1.0",
            "words": words([("clean", 0.4, 0.7), ("from", 0.7, 0.9), ("here", 0.9, 1.2),
                            ("and", 3.0, 3.2), ("to", 5.6, 5.8), ("here", 5.8, 6.1)]),
            "holds": [hold(0.2, 1.3, 800, 600, "floor", "feet", (20, 20)),
                      hold(5.3, 6.4, 2100, 1500, "floor", "feet", (20, 20))],
            "calls": [{"name": "clean", "args": {"place": {
                "from": {"deixis": "here", "word_index": 2},
                "to": {"deixis": "here", "word_index": 5}}}}],
        },
    },
    "E4-arm-put-that-here": {
        "profile": "arm-table",
        "request": {
            "version": "1.0",
            "words": words([("put", 2.0, 2.2), ("that", 2.2, 2.5), ("here", 3.4, 3.7)]),
            "holds": [hold(1.8, 2.6, 150, 250, "table", "finger", (1, 1)),
                      hold(3.2, 3.9, 480, 90, "table", "finger", (1, 1))],
            "things": BLOCKS,
            "calls": [{"name": "move", "args": {"object": {"deixis": "that", "word_index": 1},
                                                "place": {"deixis": "here", "word_index": 2}}}],
        },
    },
    "E5-arm-unsure": {
        "profile": "arm-table",
        "request": {
            "version": "1.0",
            "words": words([("put", 2.0, 2.2), ("that", 2.2, 2.5), ("here", 3.4, 3.7)]),
            "holds": [hold(1.8, 2.6, 195, 250, "table", "finger", (1, 1)),
                      hold(3.2, 3.9, 480, 90, "table", "finger", (1, 1))],
            "things": BLOCKS,
            "calls": [{"name": "move", "args": {"object": {"deixis": "that", "word_index": 1},
                                                "place": {"deixis": "here", "word_index": 2}}}],
        },
    },
    "E6-arm-a-bit-left": {
        "profile": "arm-table",
        "request": {
            "version": "1.0",
            "words": words([("move", 8.0, 8.2), ("it", 8.2, 8.3), ("a", 8.3, 8.4),
                            ("bit", 8.4, 8.6), ("left", 8.6, 8.9)]),
            "things": BLOCKS,
            "calls": [{"name": "adjust", "args": {"dx": -1}}],
            "context": {"previous": {"place": E4_PLACE}},
        },
    },
    "E7-arm-put-that-on-that": {
        "profile": "arm-table",
        "request": {
            "version": "1.0",
            "words": words([("put", 2.0, 2.2), ("that", 2.2, 2.5), ("on", 3.2, 3.4),
                            ("that", 3.4, 3.7)]),
            "holds": [hold(1.8, 2.6, 150, 250, "table", "finger", (1, 1)),
                      hold(3.1, 3.9, 325, 125, "table", "finger", (1, 1))],
            "things": BLOCKS,
            "calls": [{"name": "move", "args": {
                "object": {"deixis": "that", "word_index": 1},
                "place": {"relation": "on", "target": {"deixis": "that", "word_index": 3}}}}],
        },
    },
}  # fmt: skip


def main() -> None:
    for name, body in REQUESTS.items():
        path = HERE / f"{name}.json"
        path.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(path.name)


if __name__ == "__main__":
    main()
