"""Depth-only detection on synthetic frames: parts of every class at random poses, seen from above and at an
angle, must come out with the right class, position and yaw."""
import math

import numpy as np
from openarm_kitting_perception.detection import classify, detect_parts, PartClass
import pytest

# the kit's parts (openarm_kitting/config/kitting.yaml): a T-slot connector (long), a cube and a pinion (round)
CATALOGUE = [PartClass("connector", 0.114, 0.038, 0.039),
             PartClass("block", 0.047, 0.047, 0.047),
             PartClass("pinion", 0.032, 0.032, 0.038, is_round=True)]
SYMMETRY = {"connector": math.pi, "block": math.pi / 2}       # a part looks the same turned by this angle
WORK_Z = 0.364
REGION = [0.132, 0.328, -0.438, -0.182]
W, H, F = 1920, 1200, 737.0                      # the simulated ZED X Mini: resolution and lens
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])


def look_at(eye, target):
    """world_from_camera for an optical frame (z forward, y down) at eye looking at target."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:                     # looking straight down
        x = np.array([0, -1.0, 0])
    x /= np.linalg.norm(x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, np.cross(z, x), z, eye
    return m


def render(parts, world_from_camera):
    """Ray-cast depth of flat-topped parts on the worktop: discs for round parts, boxes for the others."""
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    rays = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones_like(u)], -1)
    d = rays @ world_from_camera[:3, :3].T
    o = world_from_camera[:3, 3]
    depth = np.full((H, W), np.inf)
    t = (WORK_Z - o[2]) / d[..., 2]                  # the worktop
    depth = np.where(t > 0, t, depth)
    for cls, x, y, yaw in parts:
        c = next(c for c in CATALOGUE if c.name == cls)
        top = WORK_Z + c.height
        for z_plane, faces in ((top, "top"),):
            t = (z_plane - o[2]) / d[..., 2]
            p = o + t[..., None] * d
            dx, dy = p[..., 0] - x, p[..., 1] - y
            # part frame: +Y is the long side, turned by yaw
            lx = dx * math.cos(yaw) + dy * math.sin(yaw)
            ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if c.is_round:
                hit = (np.hypot(lx, ly) <= c.width / 2) & (t > 0)
            else:
                hit = (np.abs(lx) <= c.width / 2) & (np.abs(ly) <= c.length / 2) & (t > 0)
            depth = np.where(hit & (t < depth), t, depth)
    return depth.astype(np.float32)                   # t along a ray with unit z component is the optical depth


# top-down over the tray, and the cell's frame camera (cell.yaml camera_frame: on the robot's centre line)
@pytest.mark.parametrize("eye, target", [((0.23, -0.31, 1.00), (0.23, -0.31, WORK_Z)),
                                         ((0.66, 0.0, 1.15), (0.08, 0.0, 0.40))], ids=["top-down", "frame"])
def test_detects_every_part(eye, target):
    rng = np.random.default_rng(7)
    parts = [("connector", 0.21, -0.30, rng.uniform(-1.5, 1.5)), ("block", 0.29, -0.22, rng.uniform(-1.5, 1.5)),
             ("pinion", 0.29, -0.39, 0.0), ("pinion", 0.16, -0.40, 0.0)]
    cam = look_at(np.array(eye), target)
    found = detect_parts(render(parts, cam), K, cam, CATALOGUE, REGION, WORK_Z, min_score=0.3)
    assert len(found) == len(parts)
    for cls, x, y, yaw in parts:
        d = min(found, key=lambda d: math.hypot(d.position[0] - x, d.position[1] - y))
        assert d.cls == cls
        assert math.hypot(d.position[0] - x, d.position[1] - y) < 0.004
        if cls in SYMMETRY:                                                       # round parts have no yaw
            period = SYMMETRY[cls]
            assert abs((d.yaw - yaw + period / 2) % period - period / 2) < math.radians(4)


def test_empty_worktop():
    cam = look_at(np.array([0.23, -0.31, 1.0]), (0.23, -0.31, WORK_Z))
    assert detect_parts(render([], cam), K, cam, CATALOGUE, REGION, WORK_Z) == []


def test_parts_outside_the_region_are_ignored():
    cam = look_at(np.array([0.23, -0.31, 1.0]), (0.23, -0.31, WORK_Z))
    found = detect_parts(render([("pinion", 0.23, -0.05, 0.0)], cam), K, cam, CATALOGUE, REGION, WORK_Z)
    assert found == []


@pytest.mark.parametrize("size, cls", [((0.0322, 0.0270, 0.0381), "pinion")])
def test_classifies_sizes_measured_in_isaac_sim(size, cls):
    """Sizes the frame camera measured in Isaac Sim: round parts look narrower across, their far side hidden."""
    name, score = classify(np.array(size), CATALOGUE)
    assert name == cls and score > 0.7
