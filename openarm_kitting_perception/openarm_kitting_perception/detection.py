"""Part detection from one depth image: segment what stands on the worktop, fit an oriented box to each
segment and name it from the part catalogue. No ROS here, so it runs in unit tests on recorded frames.

Frames: `world` has +Z up and the worktop at z = work_z. A part's frame sits at its centre with +Z up and
+Y along its long side (the catalogue's convention: box size = x, y, z with y longest; a bar's axis is y).
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PartClass:
    name: str
    length: float      # long side of the footprint
    width: float       # short side of the footprint
    height: float      # above the worktop
    is_round: bool = False     # round footprint (upright cylinder): length = width = diameter


@dataclass
class Detection:
    cls: str
    score: float               # 1 = dimensions match the catalogue exactly
    position: np.ndarray       # part centre in world
    yaw: float                 # direction of the long side (+Y of the part) is yaw + 90 deg; in (-pi/2, pi/2]
    size: np.ndarray           # measured length, width, height
    pixels: int
    mask: np.ndarray           # bool image


def catalogue_from_config(cfg):
    """Part classes from the `parts` section of kitting.yaml."""
    out = []
    for name in cfg["part_classes"]:
        spec = cfg["parts"][name]
        if spec["shape"] == "box":
            sx, sy, sz = spec["size"]
            out.append(PartClass(name, max(sx, sy), min(sx, sy), sz))
        elif spec.get("axis", "y") == "z":                       # upright cylinder
            d, length = spec["size"]
            out.append(PartClass(name, d, d, length, is_round=True))
        else:                                                    # cylinder on its side
            d, length = spec["size"]
            out.append(PartClass(name, length, d, d))
    return out


def backproject(depth, k, world_from_camera):
    """Depth image (m, optical frame) → HxWx3 points in world. Invalid pixels become NaN."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = np.where(np.isfinite(depth) & (depth > 0.05), depth, np.nan).astype(np.float32)
    cam = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], axis=-1)
    return cam @ world_from_camera[:3, :3].T.astype(np.float32) + world_from_camera[:3, 3].astype(np.float32)


def segment_above_worktop(points, region, work_z, min_height=0.006, max_height=0.10, min_pixels=120):
    """Label image of connected blobs that stand on the worktop inside region = x0, x1, y0, y1."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    with np.errstate(invalid="ignore"):
        fg = ((x > region[0]) & (x < region[1]) & (y > region[2]) & (y < region[3])
              & (z > work_z + min_height) & (z < work_z + max_height))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    masks = [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_pixels]
    return masks


def fit_part(points, mask, work_z):
    """Oriented box of one segment: centre, yaw of the long side, (length, width, height)."""
    p = points[mask]
    p = p[np.isfinite(p).all(axis=1)]
    height = float(np.percentile(p[:, 2], 95)) - work_z
    # the top face covers the whole footprint; side faces seen from the camera only add points on its edge
    (cx, cy), (a, b), angle = cv2.minAreaRect(p[:, :2].astype(np.float32))
    angle = np.radians(angle)                    # direction of side `a`
    if a < b:
        a, b, angle = b, a, angle + np.pi / 2
    yaw = angle - np.pi / 2                      # part +Y is the long side
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return np.array([cx, cy, work_z + height / 2]), float(yaw), np.array([a, b, height])


def dimension_error(size, c):
    """Largest difference between a measured size (length, width, height) and a catalogue entry.

    A round part seen at an angle hides part of its far side behind its top, so its measured footprint is a
    little narrower than the diameter in one direction; its long side is still the diameter. For round parts
    the short side only has to stay within the diameter."""
    if c.is_round:
        return max(abs(size[0] - c.length), max(0.0, size[1] - c.width), abs(size[2] - c.height))
    return max(abs(size[0] - c.length), abs(size[1] - c.width), abs(size[2] - c.height))


def classify(size, catalogue, tolerance=0.006):
    """Nearest catalogue entry by the largest dimension error; score falls to 0 at 2 x tolerance."""
    errors = [dimension_error(size, c) for c in catalogue]
    i = int(np.argmin(errors))
    return catalogue[i].name, float(max(0.0, 1.0 - errors[i] / (2 * tolerance)))


def detect_parts(depth, k, world_from_camera, catalogue, region, work_z, masks=None, min_score=0.0):
    """All parts in one depth image. `masks` replaces the depth segmentation (ground truth or a learned model)."""
    points = backproject(depth, k, world_from_camera)
    if masks is None:
        masks = segment_above_worktop(points, region, work_z)
    detections = []
    for mask in masks:
        with np.errstate(invalid="ignore"):
            mask = mask & (points[..., 2] > work_z + 0.003)      # drop mask pixels that landed on the table
        if mask.sum() < 30:
            continue
        position, yaw, size = fit_part(points, mask, work_z)
        cls, score = classify(size, catalogue)
        if score >= min_score:
            detections.append(Detection(cls, score, position, yaw, size, int(mask.sum()), mask))
    detections.sort(key=lambda d: (d.position[1], d.position[0]))
    return detections
