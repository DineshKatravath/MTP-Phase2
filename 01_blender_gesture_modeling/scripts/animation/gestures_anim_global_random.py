import bpy
import math
import random
import json
from mathutils import Euler, Vector

#  CONFIG

AXIS_X = 0  # curl
AXIS_Y = 1  # twist
AXIS_Z = 2  # spread

ARMATURE_NAME = "Armature"

FINGERS = {
    "index":  ["index_01", "index_02", "index_03"],
    "middle": ["middle_01", "middle_02", "middle_03"],
    "ring":   ["ring_01", "ring_02", "ring_03"],
    "pinky":  ["pinky_01", "pinky_02", "pinky_03"],
}

PALMS = ["palm_index","palm_middle","palm_ring","palm_pinky","palm_thumb"]

# small inward arc bias per finger for aesthetics (degrees)
BIAS = {
    "index": -4,
    "middle": -2,
    "ring": 2,
    "pinky": 4
}

# realistic limits (degrees)
LIMITS = {
    "mcp_curl_min": -20,  "mcp_curl_max": 90,
    "mcp_twist_min": -20, "mcp_twist_max": 20,
    "mcp_spread_min": -30,"mcp_spread_max": 30,
    "pip_min": 0,         "pip_max": 110,
    "dip_min": 0,         "dip_max": 100,
    "thumb_base_min_x": -30, "thumb_base_max_x": 60,
    "thumb_base_min_y": -50, "thumb_base_max_y": 50,
    "thumb_base_min_z": -50, "thumb_base_max_z": 50,
    "thumb_tip_min": 0, "thumb_tip_max": 90,
    "palm_min": -30, "palm_max": 40,
    "wrist_min": -360, "wrist_max": 360,
    "global_loc_max": 0.2  # meters (just a safety bound)
}

# dependency factors (tune for your rig)
MID_FACTOR = 0.68   # mid (PIP) is about 68% of base curl + palm influence
TIP_FACTOR = 0.6    # tip (DIP) is about 60% of mid
PALM_INFLUENCE_ON_MID = 0.25  # additional degrees added to mid from palm cupping
PALM_INFLUENCE_ON_TIP = 0.12

# animation settings
STEP = 40           # new target every STEP frames
FRAME_END = 10000

# helpers 

def deg2rad(d):
    return math.radians(d)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_tuple3(t, limits):
    return (
        clamp(t[0], limits[0], limits[1]),
        clamp(t[1], limits[2], limits[3]),
        clamp(t[2], limits[4], limits[5])
    )

# SETUP 

arm = bpy.data.objects.get(ARMATURE_NAME)
if arm is None:
    raise RuntimeError(f"Could not find object named '{ARMATURE_NAME}' in this .blend")

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones

# make armature object use quaternion rotation for stable world rotations
arm.rotation_mode = 'QUATERNION'

# clear previous animation
if arm.animation_data:
    arm.animation_data_clear()

#  CORE: applying a full DOF gesture 

def apply_gesture_full(g):
    """
    g = {
        "global": (tx, ty, tz),          # translation in meters
        "global_rot": (rx, ry, rz),      # degrees Euler to apply to arm object
        "wrist": (rx, ry, rz),           # wrist euler degrees (will be converted to quaternion)
        "palm": { palm_name: angle_deg, ... },  # 5 palms
        "fingers": {
            finger_name: {
                "base": (mcp_x, mcp_y, mcp_z),   # deg
                "mid": pip_deg,
                "tip": dip_deg
            }, ...
        },
        "thumb": { "base": (x,y,z), "tip": deg }
    }
    """
    # ---- GLOBAL / ROOT MOTION ----
    if "global" in g:
        loc = g["global"]
        # clamp small translations to avoid huge jumps
        arm.location = Vector((
            clamp(loc[0], -LIMITS["global_loc_max"], LIMITS["global_loc_max"]),
            clamp(loc[1], -LIMITS["global_loc_max"], LIMITS["global_loc_max"]),
            clamp(loc[2], -LIMITS["global_loc_max"], LIMITS["global_loc_max"]),
        ))

    if "global_rot" in g:
        gr = g["global_rot"]
        e = Euler((deg2rad(gr[0]), deg2rad(gr[1]), deg2rad(gr[2])), 'XYZ')
        arm.rotation_mode = 'QUATERNION'
        arm.rotation_quaternion = e.to_quaternion()

    # ---- WRIST (quaternion) ----
    if "wrist" in g and "wrist" in pb:
        w = pb["wrist"]
        w.rotation_mode = 'QUATERNION'
        wrist_e = g["wrist"]
        # clamp wrist angles before converting - keep safe range
        wx = clamp(wrist_e[0], LIMITS["wrist_min"], LIMITS["wrist_max"])
        wy = clamp(wrist_e[1], LIMITS["wrist_min"], LIMITS["wrist_max"])
        wz = clamp(wrist_e[2], LIMITS["wrist_min"], LIMITS["wrist_max"])
        e = Euler((deg2rad(wx), deg2rad(wy), deg2rad(wz)), 'XYZ')
        w.rotation_quaternion = e.to_quaternion()

    # ---- PALMS (1 DOF each) ----
    palm_map = g.get("palm", {})
    for p in PALMS:
        if p in pb:
            val = palm_map.get(p, 0.0)
            val = clamp(val, LIMITS["palm_min"], LIMITS["palm_max"])
            bone = pb[p]
            bone.rotation_mode = 'XYZ'
            # we keep palm as rotation on X (cup)
            bone.rotation_euler[AXIS_X] = deg2rad(val)

    # ---- FINGERS (MCP 3DOF, PIP 1DOF, DIP 1DOF) ----
    fingers_map = g.get("fingers", {})
    for fname, bones in FINGERS.items():
        finger_spec = fingers_map.get(fname, None)
        if finger_spec is None:
            # nothing provided for this finger
            continue

        base = finger_spec.get("base", (0.0, 0.0, 0.0))
        mid = finger_spec.get("mid", 0.0)
        tip = finger_spec.get("tip", 0.0)

        # clamp base (mcp) components
        bx = clamp(base[0], LIMITS["mcp_curl_min"], LIMITS["mcp_curl_max"])
        by = clamp(base[1], LIMITS["mcp_twist_min"], LIMITS["mcp_twist_max"])
        bz = clamp(base[2], LIMITS["mcp_spread_min"], LIMITS["mcp_spread_max"])

        # apply base to MCP bone (3DOF)
        mcp_name = bones[0]
        if mcp_name in pb:
            b = pb[mcp_name]
            b.rotation_mode = 'XYZ'
            # add small finger-specific bias to spread for nicer arc
            spread_bias = BIAS.get(fname, 0.0)
            b.rotation_euler = (deg2rad(bx), deg2rad(by), deg2rad(bz + spread_bias))

        # apply mid (PIP) - single axis X (curl)
        pip_name = bones[1]
        if pip_name in pb:
            bmid = pb[pip_name]
            bmid.rotation_mode = 'XYZ'
            m_val = clamp(mid, LIMITS["pip_min"], LIMITS["pip_max"])
            bmid.rotation_euler[AXIS_X] = deg2rad(m_val)

        # apply tip (DIP)
        dip_name = bones[2]
        if dip_name in pb:
            btip = pb[dip_name]
            btip.rotation_mode = 'XYZ'
            t_val = clamp(tip, LIMITS["dip_min"], LIMITS["dip_max"])
            btip.rotation_euler[AXIS_X] = deg2rad(t_val)

    # ---- THUMB (base 3DOF + tip 1DOF) ----
    thumb_spec = g.get("thumb", {})
    if thumb_spec:
        base = thumb_spec.get("base", (0.0,0.0,0.0))
        tip = clamp(thumb_spec.get("tip", 0.0), LIMITS["thumb_tip_min"], LIMITS["thumb_tip_max"])
        bx = clamp(base[0], LIMITS["thumb_base_min_x"], LIMITS["thumb_base_max_x"])
        by = clamp(base[1], LIMITS["thumb_base_min_y"], LIMITS["thumb_base_max_y"])
        bz = clamp(base[2], LIMITS["thumb_base_min_z"], LIMITS["thumb_base_max_z"])
        if "thumb_01" in pb:
            t1 = pb["thumb_01"]
            t1.rotation_mode = 'XYZ'
            t1.rotation_euler = (deg2rad(bx), deg2rad(by), deg2rad(bz))
        if "thumb_02" in pb:
            t2 = pb["thumb_02"]
            t2.rotation_mode = 'XYZ'
            t2.rotation_euler[AXIS_X] = deg2rad(tip)

#  DEPENDENCY UTILITIES 

def enforce_dependencies(g):
    """
    Given a gesture dictionary with at least 'fingers' and 'palm', compute
    PIP (mid) and DIP (tip) if they are missing, using MCP(base) and palm values.
    This mutates and returns g.
    """
    palm_map = g.get("palm", {})
    fingers_map = g.setdefault("fingers", {})

    for fname in FINGERS.keys():
        spec = fingers_map.setdefault(fname, {})
        base = spec.get("base", (0.0, 0.0, 0.0))
        # base curl is X
        base_curl = base[0]

        # palm influence: average palm cup (use the palm bone for this finger if present)
        palm_val = palm_map.get(f"palm_{fname}", None)
        if palm_val is None:
            # fallback: use palm_middle if mapping missing
            palm_val = palm_map.get("palm_middle", 0.0)

        # mid (PIP) default
        if "mid" not in spec:
            mid = base_curl * MID_FACTOR + palm_val * PALM_INFLUENCE_ON_MID
            mid = clamp(mid, LIMITS["pip_min"], LIMITS["pip_max"])
            spec["mid"] = mid

        # tip (DIP) default
        if "tip" not in spec:
            tip = spec["mid"] * TIP_FACTOR + palm_val * PALM_INFLUENCE_ON_TIP
            tip = clamp(tip, LIMITS["dip_min"], LIMITS["dip_max"])
            spec["tip"] = tip

        # clamp MCP components if user hasn't already
        bx = clamp(base[0], LIMITS["mcp_curl_min"], LIMITS["mcp_curl_max"])
        by = clamp(base[1], LIMITS["mcp_twist_min"], LIMITS["mcp_twist_max"])
        bz = clamp(base[2], LIMITS["mcp_spread_min"], LIMITS["mcp_spread_max"])
        spec["base"] = (bx, by, bz)

    return g

# RANDOM GESTURES 

def random_full_gesture():
    """
    Create a random but realistic 35-DOF gesture. MCP base values are generated,
    mid/tip are left to be computed by enforce_dependencies to ensure natural coupling.
    """
    g = {}
    # global
    g["global"] = (
        random.uniform(-0.05, 0.05),    # x
        random.uniform(-0.05, 0.05),    # y
        random.uniform(-0.02, 0.02)     # z
    )
    g["global_rot"] = (
        random.uniform(-180, 180),  # full tumble
        random.uniform(-180, 180),  # full tumble
        random.uniform(-180, 180)   # full rotation
    )

    # wrist (we will convert to quaternion in apply)
    g["wrist"] = (
        random.uniform(-80,  80),
        random.uniform(-60,  60),
        random.uniform(-180, 180)
    )

    # palm bones (small cup or relax)
    g["palm"] = {
        "palm_index": random.uniform(-6, 12),
        "palm_middle": random.uniform(-6, 18),
        "palm_ring": random.uniform(-6, 18),
        "palm_pinky": random.uniform(-8, 20),
        "palm_thumb": random.uniform(-10, 18)
    }

    # fingers - generate MCP base only; mid & tip created by enforce_dependencies
    g["fingers"] = {}
    for f in FINGERS.keys():
        base_curl = random.uniform(0, 95)  # main curl
        base_twist = random.uniform(-8, 8)
        base_spread = random.uniform(-12, 12) + BIAS.get(f, 0.0)
        g["fingers"][f] = {
            "base": (base_curl, base_twist, base_spread)
            # 'mid' and 'tip' will be added by enforce_dependencies
        }

    # thumb
    g["thumb"] = {
        "base": (
            random.uniform(-15, 35),   # curl-ish
            random.uniform(-35, 35),   # abduction (local)
            random.uniform(-30, 30)    # rotation
        ),
        # tip
        "tip": random.uniform(0, 60)
    }

    # enforce deps to fill mid/tip
    enforce_dependencies(g)
    return g

#  KEYFRAME UTILITY 

def keyframe_all():
    """
    Keyframe bones (rotation_euler or rotation_quaternion) and arm location/rotation.
    """
    for bone in pb:
        if bone.rotation_mode == 'QUATERNION':
            bone.keyframe_insert(data_path="rotation_quaternion")
        else:
            bone.keyframe_insert(data_path="rotation_euler")

    # keyframe armature object transforms
    arm.keyframe_insert(data_path="location")
    arm.keyframe_insert(data_path="rotation_quaternion")
    

# continuous realistic motion 

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = FRAME_END

# clear existing animation (safety)
if arm.animation_data:
    arm.animation_data_clear()

# frame 1 keep current scene pose (no changes)
scene.frame_set(1)
# ensure current arm object rotation is keyframed
keyframe_all()

# generate and key random targets every STEP frames
for f in range(STEP, scene.frame_end + 1, STEP):
    scene.frame_set(f)
    g = random_full_gesture()
    # you may tweak g here (or choose predefined gesture maps)
    apply_gesture_full(g)
    keyframe_all()

# return to frame 1
scene.frame_set(1)
print("Generated continuous realistic motion (full 35-DOF) with dependent PIP/DIP.")


# Logging positions in each frame

def extract_frame_data(frame):

    scene = bpy.context.scene
    scene.frame_set(frame)

    data = {}

    # -------- GLOBAL --------
    data["global"] = {
        "position": list(arm.location),
        "rotation_quaternion": list(arm.rotation_quaternion)
    }

    joints = {}

    for bone in pb:

        name = bone.name

        # ---- POSITIONS ----
        local_pos = bone.head
        world_pos = arm.matrix_world @ bone.head

        # ---- ROTATIONS ----
        if bone.rotation_mode == 'QUATERNION':
            rot_q = list(bone.rotation_quaternion)
            rot_e = list(bone.rotation_quaternion.to_euler('XYZ'))
        else:
            rot_e = list(bone.rotation_euler)
            rot_q = list(bone.rotation_euler.to_quaternion())

        joints[name] = {
            "local_position": [local_pos.x, local_pos.y, local_pos.z],
            "world_position": [world_pos.x, world_pos.y, world_pos.z],
            "rotation_euler": rot_e,
            "rotation_quaternion": rot_q
        }

    data["joints"] = joints

    return data


dataset = []
prev = None

for f in range(scene.frame_start, scene.frame_end + 1):

    frame_data = extract_frame_data(f)

    # -------- GLOBAL VELOCITY --------
    if prev:
        vel = [
            frame_data["global"]["position"][i] - prev["global"]["position"][i]
            for i in range(3)
        ]
    else:
        vel = [0.0, 0.0, 0.0]

    frame_data["global_velocity"] = vel

    # -------- JOINT VELOCITIES (VERY IMPORTANT FOR RF) --------
    if prev:
        for j in frame_data["joints"]:
            curr = frame_data["joints"][j]["world_position"]
            prev_j = prev["joints"][j]["world_position"]

            v = [curr[i] - prev_j[i] for i in range(3)]
            frame_data["joints"][j]["velocity"] = v
    else:
        for j in frame_data["joints"]:
            frame_data["joints"][j]["velocity"] = [0.0, 0.0, 0.0]

    dataset.append({
        "frame": f,
        **frame_data
    })

    prev = frame_data

# -------- SAVE --------
filepath = bpy.path.abspath("//hand_motion_full.json")

with open(filepath, "w") as file:
    json.dump(dataset, file, indent=2)

print(f"Saved full dataset to {filepath}")