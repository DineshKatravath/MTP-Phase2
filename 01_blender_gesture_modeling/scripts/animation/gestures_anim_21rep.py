import bpy
import math
import json
from mathutils import Euler

# ---------------- CONFIG ----------------

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

ARMATURE_NAME = "Armature"

FINGERS = {
    "index":  ["index_01","index_02","index_03"],
    "middle": ["middle_01","middle_02","middle_03"],
    "ring":   ["ring_01","ring_02","ring_03"],
    "pinky":  ["pinky_01","pinky_02","pinky_03"],
}

PALMS = ["palm_index","palm_middle","palm_ring","palm_pinky","palm_thumb"]

CURL_WEIGHTS = [1.0,0.7,0.4]

BIAS = {
    "index": -4,
    "middle": -2,
    "ring": 2,
    "pinky": 4
}

def deg(a):
    return math.radians(a)

# ---------------- SETUP ----------------

arm = bpy.data.objects[ARMATURE_NAME]
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')

pb = arm.pose.bones

if arm.animation_data:
    arm.animation_data_clear()

# ---------------- APPLY GESTURE ----------------

def apply_gesture(g):

    finger_curls = g["fingers"]
    spread = g["spread"]
    palm = g["palm"]
    wrist = g["wrist"]
    thumb = g["thumb"]

    # ---- FINGERS ----
    for finger,bones in FINGERS.items():

        curl = finger_curls.get(finger,0)
        spread_val = spread * (1 - curl/100)

        for i,bone_name in enumerate(bones):

            if bone_name not in pb:
                continue

            b = pb[bone_name]
            b.rotation_mode = 'XYZ'

            weight = CURL_WEIGHTS[min(i,len(CURL_WEIGHTS)-1)]

            b.rotation_euler[AXIS_X] = deg(curl * weight)

            if i == 0:
                b.rotation_euler[AXIS_Z] = deg(spread_val + BIAS.get(finger,0))

    # ---- THUMB ----
    if "thumb_01" in pb:
        t1 = pb["thumb_01"]
        t1.rotation_mode='XYZ'
        t1.rotation_euler[AXIS_X] = deg(thumb)
        t1.rotation_euler[AXIS_Y] = deg(50)
        t1.rotation_euler[AXIS_Z] = deg(70)

    if "thumb_02" in pb:
        t2 = pb["thumb_02"]
        t2.rotation_mode='XYZ'
        t2.rotation_euler[AXIS_X] = deg(thumb*0.8)

    # ---- PALM ----
    for p in PALMS:
        if p in pb:
            bone = pb[p]
            bone.rotation_mode='XYZ'
            bone.rotation_euler[AXIS_X] = deg(palm)

    # ---- WRIST ROTATION ----
    if "wrist" in pb:

        w = pb["wrist"]
        w.rotation_mode='QUATERNION'

        e = Euler((deg(wrist[0]),deg(wrist[1]),deg(wrist[2])),'XYZ')
        w.rotation_quaternion = e.to_quaternion()

    # ---- GLOBAL MOTION (IMPORTANT) ----
    if "global" in g:
        arm.location = g["global"]

# ---------------- KEYFRAME ----------------

def keyframe_all():

    for bone in pb:
        if bone.rotation_mode=='QUATERNION':
            bone.keyframe_insert(data_path="rotation_quaternion")
        else:
            bone.keyframe_insert(data_path="rotation_euler")

    arm.keyframe_insert(data_path="location")

# ---------------- GESTURES ----------------

OPEN = {
    "fingers": {"index":0,"middle":0,"ring":0,"pinky":0},
    "thumb": 0,
    "spread": 8,
    "palm": 0,
    "wrist": (-160,200,90),
    "global": (0,0,0)
}

FIST = {
    "fingers": {"index":85,"middle":90,"ring":95,"pinky":100},
    "thumb": 90,
    "spread": 10,
    "palm": 25,
    "wrist": (-160,200,90),
    "global": (-40,0,0)
}

WAVE_LEFT = {
    "fingers": {"index":0,"middle":0,"ring":0,"pinky":0},
    "thumb": 0,
    "spread": 6,
    "palm": 0,
    "wrist": (-160,200,60),
    "global": (40,0.2,0)
}

WAVE_RIGHT = {
    "fingers": {"index":0,"middle":0,"ring":0,"pinky":0},
    "thumb": 0,
    "spread": 6,
    "palm": 0,
    "wrist": (-160,200,120),
    "global": (0.1,0.2,0)
}

THUMBS_UP = {
    "fingers": {"index":90,"middle":95,"ring":95,"pinky":95},
    "thumb": -10,
    "spread": 0,
    "palm": 20,
    "wrist": (-160,180,180),
    "global": (0,0,0)
}

THUMBS_DOWN = {
    "fingers": {"index":90,"middle":95,"ring":95,"pinky":95},
    "thumb": -10,
    "spread": 0,
    "palm": 20,
    "wrist": (30,170,-330),
    "global": (0,0,0)
}

# ---------------- ANIMATION ----------------

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 240

timeline = [
    (1,OPEN),
    (30,FIST),
    (60,OPEN),
    (90,WAVE_LEFT),
    (120,WAVE_RIGHT),
    (150,WAVE_LEFT),
    (180,THUMBS_UP),
    (210,THUMBS_DOWN),
    (240,OPEN)
]

for frame,gesture in timeline:
    scene.frame_set(frame)
    apply_gesture(gesture)
    keyframe_all()

scene.frame_set(1)
bpy.context.view_layer.update()

print("Animation created")

# ============================================================
# KEYPOINT EXTRACTION
# ============================================================
BONES_TO_LOG = [
    # wrist
    "wrist",

    # palm
    "palm_thumb",
    "palm_index",
    "palm_middle",
    "palm_ring",
    "palm_pinky",

    # thumb
    "thumb_01","thumb_02",

    # fingers
    "index_01","index_02","index_03",
    "middle_01","middle_02","middle_03",
    "ring_01","ring_02","ring_03",
    "pinky_01","pinky_02","pinky_03",
]

def extract_keypoints(frame):

    scene.frame_set(frame)
    bpy.context.view_layer.update()

    keypoints = {}

    for bone_name in BONES_TO_LOG:

        if bone_name not in pb:
            continue

        bone = pb[bone_name]

        # use head for joints
        pos = arm.matrix_world @ bone.head

        keypoints[bone_name] = [
            pos.x,
            pos.y,
            pos.z
        ]

    return keypoints

# ============================================================
# DATASET
# ============================================================

dataset = []
prev = None

for f in range(scene.frame_start, scene.frame_end + 1):

    kp_global = extract_keypoints(f)

    wrist = kp_global["wrist"]

    # relative coords
    kp_relative = {
        k: [
            kp_global[k][i] - wrist[i]
            for i in range(3)
        ]
        for k in kp_global
    }

    frame_data = {
        "frame": f,
        "keypoints_global": kp_global,
        "keypoints_relative": kp_relative,
        "wrist_global": wrist
    }

    # velocity
    if prev:
        velocity = {}

        for k in kp_global:
            velocity[k] = [
                kp_global[k][i] - prev[k][i]
                for i in range(3)
            ]

        frame_data["velocity"] = velocity
    else:
        frame_data["velocity"] = {k:[0,0,0] for k in kp_global}

    dataset.append(frame_data)
    prev = kp_global

# ---------------- SAVE ----------------

filepath = bpy.path.abspath("//hand_keypoints_6dof.json")

with open(filepath, "w") as file:
    json.dump(dataset, file, indent=2)

print("Saved dataset with global motion")