"""Random part layout in the inbound zone, shared by the Isaac Sim scene and the mock-hardware runs
(plain Python + NumPy: Isaac Sim's interpreter imports it from the source tree)."""
import numpy as np


def footprint_radius(spec):
    """Radius of the circle around a part's footprint seen from above (any yaw fits inside it)."""
    if spec["shape"] == "box":
        return float(np.hypot(spec["size"][0], spec["size"][1])) / 2
    d, length = spec["size"]
    return d / 2 if spec.get("axis", "y") == "z" else float(np.hypot(d, length)) / 2


def random_layout(cfg, n_parts, seed):
    """[(class, x, y, yaw)] for up to n_parts parts with their centres in inbound_region (where the arm reaches),
    lying wholly inside inbound_area (the tray), part_gap clear of its walls and of each other at any yaw.

    Classes cycle through part_classes in a shuffled order, so every class appears once before any repeats; the
    largest parts are placed first; if no layout fits them all, the one with the most classes, then parts, is used.
    yaw (rad) turns the part's long side away from world +Y. seed 0 = different every run.
    """
    rng = np.random.default_rng(seed or None)
    x0, x1, y0, y1 = cfg["inbound_area"]
    rx0, rx1, ry0, ry1 = cfg["inbound_region"]
    gap = cfg["part_gap"]
    classes = list(cfg["part_classes"])
    rng.shuffle(classes)
    wanted = [classes[i % len(classes)] for i in range(n_parts)]
    radius = {c: footprint_radius(cfg["parts"][c]) for c in classes}

    def score(parts):
        return len({cls for _, cls, _ in parts}), len(parts)

    best = []
    for _ in range(200):                                         # whole layouts, until every part fits
        placed = []                                              # (order, class, centre)
        for order, cls in sorted(enumerate(wanted), key=lambda oc: -radius[oc[1]]):
            r = radius[cls] + gap
            lo, hi = (max(x0 + r, rx0), max(y0 + r, ry0)), (min(x1 - r, rx1), min(y1 - r, ry1))
            if lo[0] > hi[0] or lo[1] > hi[1]:
                continue
            for _ in range(200):
                c = rng.uniform(lo, hi)
                if all(np.hypot(*(c - s)) >= radius[cls] + radius[o] + gap for _, o, s in placed):
                    placed.append((order, cls, c))
                    break
        if score(placed) > score(best):                          # every class first, then as many as fit
            best = placed
        if len(best) == n_parts:
            break
    placed = sorted(best, key=lambda p: p[0])
    return [(cls, float(c[0]), float(c[1]), float(rng.uniform(-np.pi, np.pi))) for _, cls, c in placed]
