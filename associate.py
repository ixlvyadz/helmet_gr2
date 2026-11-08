"""Geometry helpers: which helmet belongs to which rider.

All boxes are (x1, y1, x2, y2) in any consistent unit (pixels or normalized).
"""
import numpy as np


def area(b):
    return max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def rider_on_motorcycle(rider, motos, thr=0.25):
    """True if enough of the rider box overlaps any motorcycle box.

    Filters out pedestrians that were detected as 'rider'.
    """
    ra = area(rider)
    return ra > 0 and any(inter(rider, m) / ra >= thr for m in motos)


def helmet_score(helmet, rider, head_frac=0.4):
    """0 if the helmet cannot belong to this rider, else a score in (0, 1].

    A helmet is valid when:
      * its centre is horizontally inside the rider box,
      * its centre is in the upper `head_frac` of the rider box
        (rejects helmets held in hand or hanging on the handlebar),
      * at least half of the helmet box lies inside the rider box.
    The score prefers helmets close to where a head normally is.
    """
    x1, y1, x2, y2 = rider
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or area(helmet) <= 0:
        return 0.0
    hx, hy = (helmet[0] + helmet[2]) / 2, (helmet[1] + helmet[3]) / 2
    if not (x1 <= hx <= x2):
        return 0.0
    if not (y1 - 0.1 * h <= hy <= y1 + head_frac * h):
        return 0.0
    containment = inter(helmet, rider) / area(helmet)
    if containment < 0.5:
        return 0.0
    ex, ey = (x1 + x2) / 2, y1 + 0.15 * h  # expected head centre
    dx, dy = abs(hx - ex) / w, abs(hy - ey) / h
    return containment * (1.0 - min(1.0, dx + dy))


def associate_helmets(riders, helmets, head_frac=0.4, min_score=0.15):
    """One-to-one greedy matching of helmets to riders by best score.

    Returns {rider_index: (helmet_index, score)}. Riders missing from the
    result have no helmet -> candidate violation.
    Because each helmet is used once, a passenger's helmet is not
    credited to the driver and vice versa.
    """
    pairs = []
    for ri, r in enumerate(riders):
        for hi, h in enumerate(helmets):
            s = helmet_score(h, r, head_frac)
            if s >= min_score:
                pairs.append((s, ri, hi))
    pairs.sort(reverse=True)
    used_r, used_h, out = set(), set(), {}
    for s, ri, hi in pairs:
        if ri in used_r or hi in used_h:
            continue
        out[ri] = (hi, s)
        used_r.add(ri)
        used_h.add(hi)
    return out
