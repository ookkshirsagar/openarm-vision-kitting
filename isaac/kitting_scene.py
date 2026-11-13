"""Isaac Sim scene for the OpenArm vision kitting demo (hardware:=isaac).

Imports the bimanual OpenArm 2.0 from the repo URDF and builds the kitting cell around it from
openarm_kitting_cell/config/cell.yaml: NVIDIA's warehouse, packing bench, pedestal and KLT bin (referenced from
the Isaac Sim asset library, never copied), plus the cell's own trays and camera frame. Drops NVIDIA Factory parts
(pinion, connector, block) at random in the inbound tray and bridges robot and cameras to ROS 2. Parts, drop region and
camera settings come from openarm_kitting/config/kitting.yaml, the same file the ROS 2 side reads.

  isaac side: isaac/run_scene.sh [--parts 6] [--seed 3] [--headless]     (expands the URDF, sets up ROS 2)
  ros2 side:  ros2 launch openarm_kitting_moveit_config moveit.launch.py hardware:=isaac

ROS 2 interface
  /joint_states                          (pub)  every joint; read by MoveIt and topic_based_ros2_control
  /joint_command                         (sub)  position targets from topic_based_ros2_control
  /clock                                 (pub)  simulation time
  /<cam>/rgb/image_rect_color            (pub)  colour image, <cam> = ext_zed (camera frame) and zed (head, colour only)
  /<cam>/rgb/camera_info                 (pub)
  /<cam>/depth/depth_registered          (pub)  32FC1 depth in metres, aligned with the colour image
  /<cam>/semantic/image, .../labels      (pub)  ground-truth class mask and its id → class table
  /<cam>/instance/image, .../labels      (pub)  ground-truth instance mask and its id → prim table
  /video/rgb                             (pub)  free viewpoint for the demo video (video_camera in the config)
  /tf                                    (pub)  world → gt/<part>: the simulated parts' poses
  /isaac_attach_cmd                      (sub)  "ATTACH <left|right> <part>" / "DETACH <part>" (isaac_attach_bridge)
"""
import argparse
import collections
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", required=True, help="xacro-expanded URDF (hardware:=isaac)")
parser.add_argument("--usd-dir", default=os.path.expanduser("~/.cache/openarm_vision_kitting/usd"),
                    help="where the URDF importer writes the robot USD")
parser.add_argument("--config", default="", help="task config (default: the installed config/kitting.yaml)")
parser.add_argument("--parts", type=int, default=6, help="number of parts dropped in the inbound region")
parser.add_argument("--seed", type=int, default=0, help="seed for the part layout (0 = different every run)")
parser.add_argument("--duration", type=float, default=0.0, help="stop after this many sim seconds (0 = run forever)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--assets", default="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac",
                    help="Isaac Sim asset root (NVIDIA's asset library; a local mirror works too)")
parser.add_argument("--dome", type=float, default=300.0, help="dome light intensity")
parser.add_argument("--work-light", type=float, default=6000.0, help="intensity of the work light over the bench")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.graph.core as og
import omni.kit.app
import omni.usd
import usdrt.Sdf
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ext = omni.kit.app.get_app().get_extension_manager()
for e in ("omni.scene.optimizer.core", "isaacsim.robot.schema", "isaacsim.asset.importer.urdf", "isaacsim.ros2.bridge"):
    ext.set_extension_enabled_immediate(e, True)
simulation_app.update()

import rclpy
from rosgraph_msgs.msg import Clock  # noqa: F401  (forces the bundled message libs to load)
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from isaacsim.asset.importer.urdf.impl import URDFImporter, URDFImporterConfig
from isaacsim.core.experimental.utils.semantics import add_labels
from isaacsim.core.simulation_manager import SimulationManager


def share(pkg):
    # resolved from AMENT_PREFIX_PATH: the `ros2` CLI would run under Isaac's Python here
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix and os.path.isdir(os.path.join(prefix, "share", pkg)):
            return os.path.join(prefix, "share", pkg)
    raise SystemExit(f"package '{pkg}' not found: source /opt/ros/jazzy and the workspace first")


cfg_path = args.config or os.path.join(share("openarm_kitting"), "config", "kitting.yaml")
task_cfg = yaml.safe_load(open(cfg_path))["/**"]["ros__parameters"]
WORK_Z = task_cfg["work_z"]

# ---------------------------------------------------------------- robot
os.makedirs(args.usd_dir, exist_ok=True)
cfg = URDFImporterConfig()
cfg.urdf_path = args.urdf
cfg.usd_path = args.usd_dir
cfg.merge_fixed_joints = False                       # keep the ee_base links as their own prims for ATTACH
cfg.fix_base = True
cfg.allow_self_collision = False
cfg.ros_package_paths = [{"name": p, "path": share(p)} for p in ("openarm_description", "zed_description", "openarm_kitting_cell")]
robot_usd = URDFImporter(cfg).import_urdf()

stage_utils.create_new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
stage_utils.add_reference_to_stage(robot_usd, "/World/robot")
simulation_app.update()


def find(root, pred):
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if pred(p):
            return p
    return None


art_root = find("/World/robot", lambda p: p.HasAPI(UsdPhysics.ArticulationRootAPI))
hands = {s: find("/World/robot", lambda p, s=s: p.GetName() == f"openarm_{s}_ee_base_link" and p.HasAPI(UsdPhysics.RigidBodyAPI))
         for s in ("left", "right")}
fingers = {s: [find("/World/robot", lambda p, n=f"openarm_{s}_ee_link{i}": p.GetName() == n and p.HasAPI(UsdPhysics.RigidBodyAPI))
               for i in (1, 2)] for s in ("left", "right")}
assert art_root and all(hands.values()), "articulation root / ee_base links not found in the imported robot"
assert all(all(f) for f in fingers.values()), "finger links not found in the imported robot"
# the importer also marks the frame camera's link as a root; PhysX rejects nested roots
for p in Usd.PrimRange(stage.GetPrimAtPath("/World/robot")):
    if p != art_root and p.HasAPI(UsdPhysics.ArticulationRootAPI):
        p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
print(f"[scene] articulation: {art_root.GetPath()}")

urdf = ET.parse(args.urdf).getroot()
urdf_joints = {j.get("name"): j for j in urdf.findall("joint")}
mimics = {n: j.find("mimic") for n, j in urdf_joints.items() if j.find("mimic") is not None}
home = yaml.safe_load(open(os.path.join(share("openarm_kitting_moveit_config"), "config/initial_positions.yaml")))
home = home["initial_positions"]
joints = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/robot")) if p.IsA(UsdPhysics.RevoluteJoint)}

# stiff position drives: the arms track MoveIt, the fingers squeeze with the URDF effort limit
n_drives = 0
for name, p in joints.items():
    if name in mimics:
        continue
    finger = "finger" in name
    # fingers squeeze with at most 2 N m (~20 N at the fingertips), the arms get the URDF effort limit
    effort = 2.0 if finger else float(urdf_joints[name].find("limit").get("effort"))
    stiffness, damping = (20.0, 2.0) if finger else (400.0, 40.0)       # per degree
    drive = UsdPhysics.DriveAPI.Apply(p, "angular")                      # USD angular units are degrees
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr().Set(stiffness)
    drive.CreateDampingAttr().Set(damping)
    drive.CreateMaxForceAttr().Set(effort)
    q0 = float(np.degrees(home.get(name, 0.0)))
    drive.CreateTargetPositionAttr().Set(q0)
    PhysxSchema.JointStateAPI.Apply(p, "angular").CreatePositionAttr().Set(q0)
    n_drives += 1

# the importer writes URDF <mimic> as NewtonMimicAPI only, which PhysX ignores: couple them with PhysxMimicJointAPI
# (q + gearing * q_ref + offset = 0)
for name, m in mimics.items():
    p, ref = joints[name], joints[m.get("joint")]
    p.RemoveAPI(UsdPhysics.DriveAPI, "angular")
    api = PhysxSchema.PhysxMimicJointAPI.Apply(p, "rotX")
    api.CreateReferenceJointRel().SetTargets([ref.GetPath()])
    api.CreateReferenceJointAxisAttr().Set("rotX")
    api.CreateGearingAttr().Set(-float(m.get("multiplier", 1)))
    api.CreateOffsetAttr().Set(float(m.get("offset", 0)))
print(f"[scene] drives on {n_drives} actuated joints, {len(mimics)} mimic joints coupled")

# gravity compensation, as the real arm's controller does it: the links carry no weight, so the position drives
# (capped at the URDF effort limit) hold the commanded pose exactly; held parts still have their weight
n_links = 0
for p in Usd.PrimRange(stage.GetPrimAtPath("/World/robot")):
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr().Set(True)
        n_links += 1
print(f"[scene] gravity compensation on {n_links} robot links")

# the importer only sets Newton's articulation flags; PhysX needs its own (self-collision is on by default)
art = PhysxSchema.PhysxArticulationAPI.Apply(art_root)
art.CreateEnabledSelfCollisionsAttr().Set(False)
art.CreateSolverPositionIterationCountAttr().Set(32)
art.CreateSolverVelocityIterationCountAttr().Set(4)        # steadier contacts between fingers and parts

# grippy finger pads
pad_mat = UsdShade.Material.Define(stage, "/World/Materials/rubber")
pm = UsdPhysics.MaterialAPI.Apply(pad_mat.GetPrim())
pm.CreateStaticFrictionAttr(2.0); pm.CreateDynamicFrictionAttr(1.5); pm.CreateRestitutionAttr(0.0)
for p in Usd.PrimRange(stage.GetPrimAtPath("/World/robot")):
    if p.HasAPI(UsdPhysics.CollisionAPI) and "ee_link" in p.GetPath().pathString:
        UsdShade.MaterialBindingAPI.Apply(p).Bind(pad_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

# ---------------------------------------------------------------- cell (cell.yaml): NVIDIA assets + own models
cell = yaml.safe_load(open(os.path.join(share("openarm_kitting_cell"), "config", "cell.yaml")))
ASSETS = args.assets.rstrip("/") + "/"
WT, FLOOR_Z = cell["worktop_z"], cell["floor_z"]
bbox_cache = lambda: UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])     # noqa: E731


def world_range(prim):
    return bbox_cache().ComputeWorldBound(prim).ComputeAlignedRange()


def asset(path, url, yaw_deg=0.0, height=None):
    """Reference an NVIDIA asset under an Xform at `path`, turned by yaw. If `height` is given and the asset comes
    in at another scale (some declare centimetres but are modelled in metres), scale it to that height."""
    xf = UsdGeom.Xform.Define(stage, path)
    inner = stage.DefinePrim(f"{path}/asset")
    inner.GetReferences().AddReference(ASSETS + url)
    simulation_app.update()
    ops = UsdGeom.Xformable(xf)
    t_op, r_op, s_op = ops.AddTranslateOp(), ops.AddRotateZOp(), ops.AddScaleOp()
    r_op.Set(yaw_deg)
    s_op.Set(Gf.Vec3f(1, 1, 1))
    if height:
        h = world_range(xf.GetPrim()).GetSize()[2]
        if h > 0 and not 0.8 < height / h < 1.25:
            s_op.Set(Gf.Vec3f(*(3 * [height / h])))
    return xf.GetPrim(), t_op


def make_static(prim):
    """Fixture: no rigid body of its own and no collision of its own (the cell's invisible boxes collide instead,
    the same boxes MoveIt plans against). A dynamic KLT bin would slide when touched."""
    for p in Usd.PrimRange(prim):
        if p.IsInstanceable():
            p.SetInstanceable(False)
    for p in Usd.PrimRange(prim):
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
            if p.HasAPI(api):
                p.RemoveAPI(api)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)


def align(prim, t_op, x=None, y=None, x0=None, top=None, bottom=None):
    """Translate so the world bounding box has centre x / y, near edge x0, and top or bottom z."""
    r = world_range(prim)
    lo, hi, c = r.GetMin(), r.GetMax(), (r.GetMin() + r.GetMax()) / 2
    d = Gf.Vec3d(0, 0, 0)
    if x is not None:
        d[0] = x - c[0]
    if x0 is not None:
        d[0] = x0 - lo[0]
    if y is not None:
        d[1] = y - c[1]
    if top is not None:
        d[2] = top - hi[2]
    if bottom is not None:
        d[2] = bottom - lo[2]
    t_op.Set(t_op.Get() + d if t_op.Get() is not None else d)


def material(name, color, metallic=0.0, roughness=0.5):
    m = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def box(path, x0, x1, y0, y1, z0, z1, look=None, collide=True, visible=True):
    """Axis-aligned box from extents: own models, and colliders matching MoveIt's collision boxes."""
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(g)
    xf.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(x1 - x0, y1 - y0, z1 - z0))
    if look is not None:
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(look)
    if collide:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    if not visible:
        g.CreateVisibilityAttr("invisible")
    return g.GetPrim()


def prism(path, x, y, z0, z1, radius, sides, look, yaw=0.0, ring=None):
    """Vertical prism (disc with many sides, hexagon with 6, or the given corners): the kit tray's pocket markings."""
    g = UsdGeom.Mesh.Define(stage, path)
    ring = ring or [(x + radius * math.cos(yaw + 2 * math.pi * k / sides),
                     y + radius * math.sin(yaw + 2 * math.pi * k / sides)) for k in range(sides)]
    g.CreatePointsAttr([Gf.Vec3f(px, py, z) for z in (z0, z1) for px, py in ring])
    faces = [list(range(sides))[::-1], list(range(sides, 2 * sides))]
    faces += [[k, (k + 1) % sides, sides + (k + 1) % sides, sides + k] for k in range(sides)]
    g.CreateFaceVertexCountsAttr([len(f) for f in faces])
    g.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    g.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(look)


steel = material("brushed_steel", (0.78, 0.79, 0.80), metallic=1.0, roughness=0.32)
alu = material("anodised_aluminium", (0.80, 0.82, 0.85), metallic=1.0, roughness=0.42)
kit_black = material("kit_tray_pom", (0.035, 0.037, 0.040), roughness=0.55)
pocket = material("kit_pocket", (0.20, 0.21, 0.22), roughness=0.75)
plate = material("powder_coat_dark", (0.10, 0.11, 0.12), metallic=0.2, roughness=0.6)
tray = material("esd_tray", (0.13, 0.15, 0.18), metallic=0.1, roughness=0.65)

# warehouse around the cell; its floor at the cell's floor height
env = cell["environment"]
env_prim, env_t = asset("/World/environment", env["asset"], math.degrees(env["yaw"]))
env_t.Set(Gf.Vec3d(env["offset"][0], env["offset"][1], FLOOR_Z))
floor = box("/World/cell/floor_collider", -10, 10, -10, 10, FLOOR_Z - 0.05, FLOOR_Z, visible=False)

# pedestal under the robot, adapter plate on top
pd = cell["pedestal"]
stand, stand_t = asset("/World/cell/pedestal", pd["asset"], height=-FLOOR_Z - pd["plate_t"])
make_static(stand)
align(stand, stand_t, x=pd["xy"][0], y=pd["xy"][1], top=-pd["plate_t"])
pp = pd["plate"]
box("/World/cell/pedestal_plate", pp[0], pp[1], pp[2], pp[3], -pd["plate_t"], 0.0, plate)

# packing bench, long side to the robot; its own tote on the top is switched off (the cell brings its trays)
bn = cell["bench"]
bench, bench_t = asset("/World/cell/bench", bn["asset"], 90.0)
make_static(bench)
for name in bn.get("hide", []):
    for p in Usd.PrimRange(bench):
        if p.GetName() == name:
            p.SetActive(False)
simulation_app.update()
r = world_range(bench)
if not 0.8 < (WT - FLOOR_Z) / r.GetSize()[2] < 1.25:     # modelled in metres, declared in centimetres
    k = (WT - FLOOR_Z) / r.GetSize()[2]
    UsdGeom.Xformable(bench).GetOrderedXformOps()[2].Set(Gf.Vec3f(k, k, k))
e = bn["extent"]
align(bench, bench_t, x0=e[0], y=(e[2] + e[3]) / 2, top=WT)
box("/World/cell/bench_top_collider", e[0], e[1], e[2], e[3], WT - bn["top_t"], WT, visible=False)

# inbound tray: brushed stainless, floor and four walls
t, tw, th, tf = (cell["inbound_tray"][k] for k in ("extent", "wall", "height", "floor"))
box("/World/cell/inbound_tray/floor", t[0], t[1], t[2], t[3], WT, WT + tf, tray)
box("/World/cell/inbound_tray/wall_back", t[0], t[0] + tw, t[2], t[3], WT, WT + th, tray)
box("/World/cell/inbound_tray/wall_front", t[1] - tw, t[1], t[2], t[3], WT, WT + th, tray)
box("/World/cell/inbound_tray/wall_right", t[0] + tw, t[1] - tw, t[2], t[2] + tw, WT, WT + th, tray)
box("/World/cell/inbound_tray/wall_left", t[0] + tw, t[1] - tw, t[3] - tw, t[3], WT, WT + th, tray)

# kit tray: black POM plate with a marked pocket per part type (round or rectangular, as the part)
k = cell["kit_tray"]["extent"]
KZ = WT + cell["kit_tray"]["height"]
box("/World/cell/kit_tray/plate", k[0], k[1], k[2], k[3], WT, KZ, kit_black)
for cls, (sx, sy, syaw) in task_cfg["kit_slots"].items():
    spec = task_cfg["parts"][cls]
    if spec["shape"] == "box":                       # rectangle, the long side along the slot's yaw (0 = world Y)
        hw, hl = min(spec["size"][:2]) / 2 + 0.005, max(spec["size"][:2]) / 2 + 0.005
        c, s_ = math.cos(syaw), math.sin(syaw)
        corners = [(sx + u * c - v * s_, sy + u * s_ + v * c) for u, v in ((hw, hl), (-hw, hl), (-hw, -hl), (hw, -hl))]
        prism(f"/World/cell/kit_tray/pocket_{cls}", 0, 0, KZ, KZ + 0.0006, 0, 4, pocket, ring=corners)
    else:
        prism(f"/World/cell/kit_tray/pocket_{cls}", sx, sy, KZ, KZ + 0.0006, spec["size"][0] / 2 + 0.005, 48, pocket)

# reject bin: NVIDIA's small load carrier; colliders matching MoveIt's (floor and four walls)
rb = cell["reject_bin"]
(rcx, rcy), (rsx, rsy, rsz), rw = rb["centre"], rb["size"], rb["wall"]
klt, klt_t = asset("/World/cell/reject_bin", rb["asset"], math.degrees(rb["yaw"]), height=rsz)
make_static(klt)
align(klt, klt_t, x=rcx, y=rcy, bottom=WT)
rx0, rx1, ry0, ry1 = rcx - rsx / 2, rcx + rsx / 2, rcy - rsy / 2, rcy + rsy / 2
for name, ext in (("floor", (rx0, rx1, ry0, ry1, WT, WT + rw)), ("back", (rx0, rx0 + rw, ry0, ry1, WT, WT + rsz)),
                  ("front", (rx1 - rw, rx1, ry0, ry1, WT, WT + rsz)), ("right", (rx0, rx1, ry0, ry0 + rw, WT, WT + rsz)),
                  ("left", (rx0, rx1, ry1 - rw, ry1, WT, WT + rsz))):
    box(f"/World/cell/reject_bin_colliders/{name}", *ext, visible=False)

# camera frame: four-post profile cage on the floor, top frame, a crossbeam over the tray carrying the camera
m = cell["camera_frame"]
mp, (fx0, fx1), (fy0, fy1), tz, eye = m["profile"], m["posts_x"], m["posts_y"], m["top_z"], m["camera_eye"]
h = mp / 2
for i, (px, py) in enumerate(((fx0, fy0), (fx0, fy1), (fx1, fy0), (fx1, fy1))):
    box(f"/World/cell/frame/post_{i}", px - h, px + h, py - h, py + h, FLOOR_Z, tz + h, alu)
    box(f"/World/cell/frame/foot_{i}", px - 0.06, px + 0.06, py - 0.06, py + 0.06, FLOOR_Z, FLOOR_Z + 0.008, plate)
for name, ext in (("back", (fx0 - h, fx0 + h, fy0, fy1)), ("front", (fx1 - h, fx1 + h, fy0, fy1)),
                  ("right", (fx0, fx1, fy0 - h, fy0 + h)), ("left", (fx0, fx1, fy1 - h, fy1 + h)),
                  ("camera_beam", (fx0, fx1, m["camera_beam_y"] - h, m["camera_beam_y"] + h))):
    box(f"/World/cell/frame/{name}", *ext, tz - h, tz + h, alu)
box("/World/cell/frame/camera_drop", eye[0] - 0.02, eye[0] + 0.02, eye[1] - 0.02, eye[1] + 0.02, eye[2] + 0.03, tz - h, alu)

# lighting: the warehouse's own lights, a soft dome and a work light over the bench
UsdLux.DomeLight.Define(stage, "/World/lights/dome").CreateIntensityAttr(args.dome)
work = UsdLux.RectLight.Define(stage, "/World/lights/work_light")
work.CreateWidthAttr(1.2); work.CreateHeightAttr(0.6); work.CreateIntensityAttr(args.work_light)
UsdGeom.Xformable(work).AddTranslateOp().Set(Gf.Vec3d(0.45, -0.2, m["top_z"] - 0.03))   # LED panel under the top frame
print(f"[scene] cell: bench, pedestal, trays, reject bin, camera frame; environment {env['asset']}", flush=True)

# ---------------------------------------------------------------- parts: NVIDIA Factory assets
# random class, position and yaw in the inbound tray (the same layout as mock hardware for a seed)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openarm_kitting"))
from openarm_kitting.layout import random_layout  # noqa: E402

layout = random_layout(task_cfg, args.parts, args.seed)
if len(layout) < args.parts:
    print(f"[scene] only {len(layout)} of {args.parts} parts fit in the inbound region", flush=True)

part_mat = UsdShade.Material.Define(stage, "/World/Materials/part")
cm = UsdPhysics.MaterialAPI.Apply(part_mat.GetPrim())
cm.CreateStaticFrictionAttr(0.9); cm.CreateDynamicFrictionAttr(0.8); cm.CreateRestitutionAttr(0.0)
UsdGeom.Xform.Define(stage, "/World/parts")
parts = {}
for i, (cls, cx, cy, yaw) in enumerate(layout):
    spec = task_cfg["parts"][cls]
    name = f"{cls}_{i}"
    root = UsdGeom.Xform.Define(stage, f"/World/parts/{name}")
    offset = UsdGeom.Xform.Define(stage, f"/World/parts/{name}/centred")
    inner = stage.DefinePrim(f"/World/parts/{name}/centred/asset")
    inner.GetReferences().AddReference(ASSETS + spec["asset"])
    simulation_app.update()
    # make the referenced model editable and a single rigid body: no instancing, no fixed joint to the world,
    # no rigid bodies of its own; convex-hull colliders
    changed = True
    while changed:
        changed = False
        for p in Usd.PrimRange(inner):
            if p.IsInstance():
                p.SetInstanceable(False)
                changed = True
    for p in list(Usd.PrimRange(inner)):
        if p.IsA(UsdPhysics.Joint):
            p.SetActive(False)
            continue
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if p.HasAPI(UsdPhysics.MassAPI):
            p.RemoveAPI(UsdPhysics.MassAPI)
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Mesh):
            if p.HasAPI(PhysxSchema.PhysxSDFMeshCollisionAPI):
                p.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set("convexHull")
    # centre the model on its bounding box: the part frame sits at the part's centre, like the detections
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeLocalBound(inner).ComputeAlignedRange()
    c = (r.GetMin() + r.GetMax()) / 2
    offset.AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -c[2]))
    xf = UsdGeom.Xformable(root)
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, WORK_Z + r.GetSize()[2] / 2 + 0.002))
    xf.AddRotateZOp().Set(math.degrees(yaw))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(spec["mass"])
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    prb.CreateAngularDampingAttr(1.0)
    prb.CreateMaxDepenetrationVelocityAttr(0.3)       # m/s: a part wedged between fingers is eased out, not fired out
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(part_mat, UsdShade.Tokens.strongerThanDescendants, "physics")
    add_labels(prim, labels=[cls], taxonomy="class")
    parts[name] = prim
print(f"[scene] parts: {', '.join(f'{n} at {x:.2f},{y:.2f}' for n, (_, x, y, _) in zip(parts, layout))}", flush=True)


def world_pose(prim):
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


# ---------------------------------------------------------------- cameras
def rpy_matrix(xyz, rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    m = np.eye(4)
    m[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    m[:3, 3] = xyz
    return m


def urdf_frame(link):
    """world → link, for links on a chain of fixed joints (the cameras)."""
    m = np.eye(4)
    by_child = {j.find("child").get("link"): j for j in urdf_joints.values()}
    while link in by_child:
        j = by_child[link]
        assert j.get("type") == "fixed", f"{j.get('name')} is not fixed"
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
        m = rpy_matrix(xyz, rpy) @ m
        link = j.find("parent").get("link")
    return m


CAM_W, CAM_H = task_cfg["camera_resolution"]
APERTURE = 20.955                                    # mm, USD default; the focal length sets the field of view
FOCAL = APERTURE / (2 * math.tan(math.radians(task_cfg["camera_hfov_deg"]) / 2))
OPTICAL_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])     # ROS optical (z forward, y down) → USD camera (-z forward, y up)
urdf_links = {l.get("name") for l in urdf.findall("link")}
UsdGeom.Xform.Define(stage, "/World/cameras")
cameras = {}
for cam in task_cfg["cameras"] + task_cfg["view_cameras"]:
    frame = f"{cam}_left_camera_frame_optical"
    if frame not in urdf_links:
        print(f"[scene] camera {cam}: no {frame} in the URDF, skipped", flush=True)
        continue
    c = UsdGeom.Camera.Define(stage, f"/World/cameras/{cam}")
    c.CreateFocalLengthAttr(FOCAL)
    c.CreateHorizontalApertureAttr(APERTURE)
    c.CreateVerticalApertureAttr(APERTURE * CAM_H / CAM_W)
    c.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
    m = urdf_frame(frame) @ OPTICAL_TO_USD
    UsdGeom.Xformable(c).AddTransformOp().Set(Gf.Matrix4d(*m.T.flatten().tolist()))    # USD matrices are row-vector
    cameras[cam] = (str(c.GetPath()), frame)

# ---------------------------------------------------------------- ROS 2 graph
og.Controller.edit(
    {"graph_path": "/World/ros2_bridge", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("tick", "omni.graph.action.OnPlaybackTick"),
            ("sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("ctx", "isaacsim.ros2.bridge.ROS2Context"),
            ("read_js", "isaacsim.sensors.physics.IsaacReadJointState"),
            ("pub_js", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        og.Controller.Keys.CONNECT: [
            ("tick.outputs:tick", "read_js.inputs:execIn"),
            ("read_js.outputs:execOut", "pub_js.inputs:execIn"),
            ("read_js.outputs:jointNames", "pub_js.inputs:jointNames"),
            ("read_js.outputs:jointPositions", "pub_js.inputs:jointPositions"),
            ("read_js.outputs:jointVelocities", "pub_js.inputs:jointVelocities"),
            ("read_js.outputs:jointEfforts", "pub_js.inputs:jointEfforts"),
            ("read_js.outputs:jointDofTypes", "pub_js.inputs:jointDofTypes"),
            ("read_js.outputs:stageMetersPerUnit", "pub_js.inputs:stageMetersPerUnit"),
            ("read_js.outputs:sensorTime", "pub_js.inputs:sensorTime"),
            ("tick.outputs:tick", "clock.inputs:execIn"),
            ("ctx.outputs:context", "pub_js.inputs:context"),
            ("ctx.outputs:context", "clock.inputs:context"),
            ("sim_time.outputs:simulationTime", "clock.inputs:timeStamp"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("read_js.inputs:prim", [usdrt.Sdf.Path(str(art_root.GetPath()))]),
            ("pub_js.inputs:topicName", "isaac/joint_states_raw"),     # republished below as /joint_states
        ],
    },
)

CAM_SKIP = 30 // int(task_cfg["camera_rate"]) - 1     # frames skipped between published images (FRAME_HZ = 30)
STREAMS = [                                          # helper type, topic, publishes a label table
    ("rgb", "rgb/image_rect_color", False),
    ("depth", "depth/depth_registered", False),
    ("semantic_segmentation", "semantic/image", True),
    ("instance_segmentation", "instance/image", True),
]
for cam, (prim_path, frame) in cameras.items():
    view_only = cam in task_cfg["view_cameras"]          # colour at half resolution, for watching only
    width, height = (CAM_W // 2, CAM_H // 2) if view_only else (CAM_W, CAM_H)
    nodes = [("tick", "omni.graph.action.OnPlaybackTick"),
             ("rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
             ("info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper")]
    connect = [("tick.outputs:tick", "rp.inputs:execIn"),
               ("rp.outputs:execOut", "info.inputs:execIn"),
               ("rp.outputs:renderProductPath", "info.inputs:renderProductPath")]
    values = [("rp.inputs:cameraPrim", [usdrt.Sdf.Path(prim_path)]),
              ("rp.inputs:width", width), ("rp.inputs:height", height),
              ("info.inputs:frameId", frame), ("info.inputs:topicName", f"{cam}/rgb/camera_info"),
              ("info.inputs:frameSkipCount", CAM_SKIP)]
    for kind, topic, labels in STREAMS[:1] if view_only else STREAMS:
        nodes.append((kind, "isaacsim.ros2.bridge.ROS2CameraHelper"))
        connect += [("rp.outputs:execOut", f"{kind}.inputs:execIn"),
                    ("rp.outputs:renderProductPath", f"{kind}.inputs:renderProductPath")]
        values += [(f"{kind}.inputs:type", kind), (f"{kind}.inputs:frameId", frame),
                   (f"{kind}.inputs:topicName", f"{cam}/{topic}"), (f"{kind}.inputs:frameSkipCount", CAM_SKIP)]
        if labels:
            values += [(f"{kind}.inputs:enableSemanticLabels", True),
                       (f"{kind}.inputs:semanticLabelsTopicName", f"{cam}/{topic.split('/')[0]}/labels")]
    og.Controller.edit({"graph_path": f"/World/camera_{cam}", "evaluator_name": "execution"},
                       {og.Controller.Keys.CREATE_NODES: nodes, og.Controller.Keys.CONNECT: connect,
                        og.Controller.Keys.SET_VALUES: values})
print(f"[scene] cameras: {', '.join(cameras)} at {CAM_W}x{CAM_H}, focal {FOCAL:.2f} mm", flush=True)

# video camera: a free viewpoint for the demo recording (/video/rgb), colour only
if task_cfg.get("video_camera"):
    eye, target = np.array(task_cfg["video_camera"][:3]), np.array(task_cfg["video_camera"][3:6])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -fwd, eye        # USD camera looks along -z, y up
    vc = UsdGeom.Camera.Define(stage, "/World/cameras/video")
    vc.CreateFocalLengthAttr(18.0); vc.CreateHorizontalApertureAttr(APERTURE)
    vc.CreateVerticalApertureAttr(APERTURE * 9 / 16); vc.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
    UsdGeom.Xformable(vc).AddTransformOp().Set(Gf.Matrix4d(*m.T.flatten().tolist()))
    vw, vh = task_cfg["video_resolution"]
    og.Controller.edit({"graph_path": "/World/camera_video", "evaluator_name": "execution"}, {
        og.Controller.Keys.CREATE_NODES: [("tick", "omni.graph.action.OnPlaybackTick"),
                                          ("rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                                          ("rgb", "isaacsim.ros2.bridge.ROS2CameraHelper")],
        og.Controller.Keys.CONNECT: [("tick.outputs:tick", "rp.inputs:execIn"),
                                     ("rp.outputs:execOut", "rgb.inputs:execIn"),
                                     ("rp.outputs:renderProductPath", "rgb.inputs:renderProductPath")],
        og.Controller.Keys.SET_VALUES: [("rp.inputs:cameraPrim", [usdrt.Sdf.Path("/World/cameras/video")]),
                                        ("rp.inputs:width", vw), ("rp.inputs:height", vh),
                                        ("rgb.inputs:type", "rgb"), ("rgb.inputs:frameId", "video"),
                                        ("rgb.inputs:topicName", "video/rgb"),
                                        ("rgb.inputs:frameSkipCount", 30 // task_cfg["video_rate"] - 1)]})
    print(f"[scene] video camera at {eye.round(2).tolist()} → {target.round(2).tolist()}, {vw}x{vh}", flush=True)

# ---------------------------------------------------------------- commands, attach + ground-truth TF (rclpy)
rclpy.init()
node = rclpy.create_node("isaac_kitting_scene")
drive_targets = {n: UsdPhysics.DriveAPI(p, "angular").GetTargetPositionAttr()
                 for n, p in joints.items() if n not in mimics}


def on_joint_command(msg):
    # topic_based_ros2_control lists the mimic joints after the commanded ones, without a position:
    # Isaac's own JointState subscriber rejects that message, so the drive targets are set here
    for name, q in zip(msg.name, msg.position):
        if name in finger_hold and abs(q) < abs(finger_hold[name]):
            q = finger_hold[name]                    # holding a part: no further than where they closed on it
        if name in drive_targets and math.isfinite(q):
            drive_targets[name].Set(math.degrees(q))


finger_now, finger_hold = {}, {}                    # finger joint → measured position; → position while holding
node.create_subscription(JointState, "/joint_command", on_joint_command, 10)
js_pub = node.create_publisher(JointState, "/joint_states", 10)
last_finger = {}                                     # finger joint → recent (stamp, position) samples
mimic_rule = {n: (m.get("joint"), float(m.get("multiplier", 1)), float(m.get("offset", 0))) for n, m in mimics.items()}


FINGER_WINDOW = 0.25                                 # s, finger velocity estimate


def on_raw_joint_states(msg):
    # PhysX holds a mimic joint to its reference only within solver tolerance (~2e-4 rad); MoveIt computes it exactly
    # from the URDF, and MTC's Connect refuses a start state that differs from that. Publish the URDF value.
    index = {n: i for i, n in enumerate(msg.name)}
    pos, vel = list(msg.position), list(msg.velocity)
    for name, (ref, k, offset) in mimic_rule.items():
        if name in index and ref in index:
            pos[index[name]] = k * pos[index[ref]] + offset
            if vel:
                vel[index[name]] = k * vel[index[ref]]
    # a finger pressed against a part keeps a constant solver velocity in PhysX (~1.3 rad/s) while its position does
    # not change, so the gripper controller would never see it stall: report the fingers' velocity from positions
    # over a 0.25 s window: a finger squeezing a part jitters by a few mrad from frame to frame
    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    if vel:
        for name, i in index.items():
            if "finger" in name:
                hist = last_finger.setdefault(name, collections.deque())
                hist.append((t, pos[i]))
                while len(hist) > 2 and t - hist[1][0] >= FINGER_WINDOW:
                    hist.popleft()
                t0, q0 = hist[0]
                vel[i] = (pos[i] - q0) / (t - t0) if t > t0 else 0.0
    msg.position, msg.velocity = pos, vel
    finger_now.update((n, pos[i]) for n, i in index.items() if n.endswith("finger_joint1"))
    js_pub.publish(msg)


node.create_subscription(JointState, "/isaac/joint_states_raw", on_raw_joint_states, 10)
tf_pub = node.create_publisher(TFMessage, "/tf", 10)
grasp_joints = {}                                    # part name → fixed joint path
grasp_arm = {}                                       # part name → finger joint of the hand holding it
part_mass = {}                                       # part name → its mass while it is held
unfilter = []                                        # (wall time, part): finger contact filter to remove


def on_attach(msg):
    words = msg.data.split()
    verb, name = words[0], words[-1]
    part = parts.get(name)
    if part is None:
        node.get_logger().warn(f"{msg.data}: no part named {name}")
        return
    if verb == "ATTACH" and len(words) == 3 and words[1] in hands and name not in grasp_joints:
        hand = hands[words[1]]
        # held upright: the rotary fingers tip a part a few degrees as they close on it, where flat pads would
        # square it; welded tipped, a cube's low corner meets the kit tray before the arm reaches the place pose
        pw = world_pose(part).RemoveScaleShear()
        xa = pw.TransformDir(Gf.Vec3d(1, 0, 0))
        upright = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(math.atan2(xa[1], xa[0]))))
        upright.SetTranslateOnly(pw.ExtractTranslation())
        rel = upright * world_pose(hand).GetInverse()       # part in hand frame
        path = Sdf.Path(f"/World/grasp_{name}")
        j = UsdPhysics.FixedJoint.Define(stage, path)
        j.CreateExcludeFromArticulationAttr(True)           # a joint to a free body, not a new link of the robot
        j.CreateBody0Rel().SetTargets([hand.GetPath()])
        j.CreateBody1Rel().SetTargets([part.GetPath()])
        t, r = rel.ExtractTranslation(), rel.ExtractRotationQuat()
        j.CreateLocalPos0Attr(Gf.Vec3f(t)); j.CreateLocalRot0Attr(Gf.Quatf(r))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0)); j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        grasp_joints[name] = path
        payload(part, True)
        # the gripper controller keeps driving the fingers shut; against a part welded to the hand that squeeze
        # goes round a loop the solver does not close exactly and twists the wrist off its target: the fingers
        # stay where they closed on the part, as a gripper holding position does, and stop pushing on it
        finger = f"openarm_{words[1]}_finger_joint1"
        if finger in finger_now:
            finger_hold[finger] = finger_now[finger]
            drive_targets[finger].Set(math.degrees(finger_now[finger]))
        pairs = UsdPhysics.FilteredPairsAPI.Apply(part).CreateFilteredPairsRel()
        for f in fingers[words[1]]:
            pairs.AddTarget(f.GetPath())
        grasp_arm[name] = finger
        node.get_logger().info(f"{name} held by the {words[1]} hand")
    elif verb == "DETACH" and name in grasp_joints:
        stage.RemovePrim(grasp_joints.pop(name))
        payload(part, False)
        finger_hold.pop(grasp_arm.pop(name, ""), None)
        unfilter.append((time.monotonic() + 2.0, part))  # fingers touch it again once they have opened
        UsdPhysics.RigidBodyAPI(part).CreateVelocityAttr().Set(Gf.Vec3f(0, 0, 0))    # wakes it: it was asleep
        node.get_logger().info(f"{name} released")


def payload(part, held):
    """Payload compensation, like the links' gravity compensation: the controller carries a held part's weight, so
    it does not pull the wrist drives off their targets."""
    PhysxSchema.PhysxRigidBodyAPI.Apply(part).CreateDisableGravityAttr().Set(held)
    mass = UsdPhysics.MassAPI(part).GetMassAttr()
    if held:
        part_mass[part.GetName()] = mass.Get()
        mass.Set(0.01)                                      # kg: nor does its inertia swing the wrist
    elif part.GetName() in part_mass:
        mass.Set(part_mass.pop(part.GetName()))


node.create_subscription(String, "/isaac_attach_cmd", on_attach, 10)


def publish_part_tf(sim_t):
    while unfilter and unfilter[0][0] < time.monotonic():
        UsdPhysics.FilteredPairsAPI(unfilter.pop(0)[1]).GetFilteredPairsRel().ClearTargets(True)
    msg = TFMessage()
    for name, prim in parts.items():
        m = world_pose(prim).RemoveScaleShear()
        t, q = m.ExtractTranslation(), m.ExtractRotationQuat()
        ts = TransformStamped()
        ts.header.stamp.sec = int(sim_t); ts.header.stamp.nanosec = int((sim_t % 1) * 1e9)
        ts.header.frame_id, ts.child_frame_id = "world", f"gt/{name}"
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = t
        ts.transform.rotation.w = q.GetReal()
        ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z = q.GetImaginary()
        msg.transforms.append(ts)
    tf_pub.publish(msg)


# ---------------------------------------------------------------- run
PHYSICS_HZ = 120
FRAME_HZ = 30                                        # app updates (render, ROS 2 graph); physics substeps in between
SimulationManager.setup_simulation(dt=1.0 / FRAME_HZ)
for p in stage.Traverse():
    if p.IsA(UsdPhysics.Scene):
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(p)
        scene_api.CreateSolverTypeAttr().Set("TGS")
        scene_api.CreateTimeStepsPerSecondAttr().Set(PHYSICS_HZ)
app_utils.play()
simulation_app.update()

step = 0
print("[scene] running — start the ROS 2 side now", flush=True)
wall0 = time.monotonic()
try:
    while simulation_app.is_running():
        simulation_app.update()
        step += 1
        sim_t = step / FRAME_HZ
        lag = sim_t - (time.monotonic() - wall0)      # never run ahead of real time: MoveIt plans on the wall clock
        if lag > 0:
            time.sleep(lag)
        for _ in range(6):                           # joint states, commands, attach: one callback per spin_once
            rclpy.spin_once(node, timeout_sec=0.0)
        if step % (FRAME_HZ * 5) == 0:
            print(f"[scene] sim {sim_t:6.1f}s  wall {time.monotonic() - wall0:6.1f}s", flush=True)
        publish_part_tf(sim_t)
        if args.duration and sim_t >= args.duration:
            break
except KeyboardInterrupt:
    pass

node.destroy_node(); rclpy.shutdown()
app_utils.stop()
simulation_app.close()
