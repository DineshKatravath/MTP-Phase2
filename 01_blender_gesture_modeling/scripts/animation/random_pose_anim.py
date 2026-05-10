import bpy
import math
import random
from mathutils import Euler

# ---------------- CONFIG ----------------

AXIS_X = 0  # Curl
AXIS_Y = 1  # Twist
AXIS_Z = 2  # Spread

ARMATURE_NAME = "Armature"

FINGERS = {
    "index":  ["index_01", "index_02", "index_03"],
    "middle": ["middle_01", "middle_02", "middle_03"],
    "ring":   ["ring_01", "ring_02", "ring_03"],
    "pinky":  ["pinky_01", "pinky_02", "pinky_03"],
}

PALMS = ["palm_index","palm_middle","palm_ring","palm_pinky","palm_thumb"]

CURL_WEIGHTS = [1.0, 0.7, 0.4]

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

# Clear old animation
if arm.animation_data:
    arm.animation_data_clear()

# ---------------- APPLY GESTURE ----------------

def apply_gesture(gesture):

    finger_curls = gesture["fingers"]
    spread = gesture["spread"]
    palm_cup = gesture["palm"]
    wrist_rot = gesture["wrist"]
    thumb_curl = gesture["thumb"]

    # ---- FINGERS ----
    for finger, bones in FINGERS.items():

        curl_value = finger_curls.get(finger, 0)
        spread_value = spread * (1 - curl_value / 100)

        for i, bone_name in enumerate(bones):
            if bone_name not in pb:
                continue

            b = pb[bone_name]
            b.rotation_mode = 'XYZ'
            weight = CURL_WEIGHTS[min(i, len(CURL_WEIGHTS)-1)]

            # Curl
            b.rotation_euler[AXIS_X] = deg(curl_value * weight)

            # Spread only base
            if i == 0:
                b.rotation_euler[AXIS_Z] = deg(spread_value + BIAS.get(finger, 0))

    # ---- THUMB ----
    thumb_curl = gesture["thumb"]

    if "thumb_01" in pb:
        t1 = pb["thumb_01"]
        t1.rotation_mode = 'XYZ'

        # Very small curl
        t1.rotation_euler[AXIS_X] = deg(thumb_curl)
        t1.rotation_euler[AXIS_Y] = deg(0)
        t1.rotation_euler[AXIS_Z] = deg(70)

    if "thumb_02" in pb:
        t2 = pb["thumb_02"]
        t2.rotation_mode = 'XYZ'

        # Keep tip almost straight
        t2.rotation_euler[AXIS_X] = deg(thumb_curl * 0.2)

    # ---- PALM (1 DOF each) ----
    for p in PALMS:
        if p in pb:
            bone = pb[p]
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler[AXIS_X] = deg(palm_cup)

    # ---- WRIST ----
    if "wrist" in pb:
        w = pb["wrist"]
        w.rotation_mode = 'QUATERNION'

        eul = Euler((
            deg(wrist_rot[0]),
            deg(wrist_rot[1]),
            deg(wrist_rot[2])
        ), 'XYZ')

        w.rotation_quaternion = eul.to_quaternion()


def keyframe_all():
    for bone in pb:
        if bone.rotation_mode == 'QUATERNION':
            bone.keyframe_insert(data_path="rotation_quaternion")
        else:
            bone.keyframe_insert(data_path="rotation_euler")

# ---------------- GESTURES----------------

OPEN = {
    "fingers": {"index":0,"middle":0,"ring":0,"pinky":0},
    "thumb": 0,
    "spread": 8,
    "palm": 0,
    "wrist": (0,0,0)
}

FIST = {
    "fingers": {"index":85,"middle":90,"ring":95,"pinky":100},
    "thumb": 70,
    "spread": 0,
    "palm": 25,
    "wrist": (0,0,0)
}

THUMBS_UP = {
    "fingers": {"index":90,"middle":95,"ring":95,"pinky":95},
    "thumb": -10,
    "spread": 0,
    "palm": 20,
    "wrist": (0, 0, 90)
}

THUMBS_DOWN = {
    "fingers": {"index":90,"middle":95,"ring":95,"pinky":95},
    "thumb": -10,
    "spread": 0,
    "palm": 20,
    "wrist": (-180,0, -90)  # rotate whole hand
}

WAVE_LEFT = {
    "fingers": {"index":0,"middle":0,"ring":0,"pinky":0},
    "thumb": 0,
    "spread": 6,
    "palm": 0,
    "wrist": (0,0,-30)
}

WAVE_RIGHT = {
    "fingers": {"index":0,"middle":0,"ring":0,"pinky":0},
    "thumb": 0,
    "spread": 6,
    "palm": 0,
    "wrist": (0,0,30)
}

# Random Gesture

def random_gesture():

    # Random finger curls
    finger_vals = {
        "index":  random.uniform(0, 100),
        "middle": random.uniform(0, 100),
        "ring":   random.uniform(0, 100),
        "pinky":  random.uniform(0, 100),
    }

    # Random thumb curl
    thumb_val = random.uniform(-20, 80)

    # Random spread
    spread_val = random.uniform(-10, 10)

    # Random palm cup
    palm_val = random.uniform(0, 30)

    # Random wrist orientation
    wrist_val = (
        random.uniform(-90, 90),
        random.uniform(-90, 90),
        random.uniform(-180, 180)
    )

    return {
        "fingers": finger_vals,
        "thumb": thumb_val,
        "spread": spread_val,
        "palm": palm_val,
        "wrist": wrist_val
    }

def random_gesture():
    finger_vals = {
        "index":  random.uniform(-20, 100),
        "middle": random.uniform(-20, 100),
        "ring":   random.uniform(-20, 100),
        "pinky":  random.uniform(-20, 100),
    }

    thumb_val = random.uniform(-20, 80)
    spread_val = random.uniform(-30, 30)
    palm_val = random.uniform(-20, 30)

    wrist_val = (
        random.uniform(-90, 90),
        random.uniform(-90, 90),
        random.uniform(-180, 180)
    )

    return {
        "fingers": finger_vals,
        "thumb": thumb_val,
        "spread": spread_val,
        "palm": palm_val,
        "wrist": wrist_val
    }


# ---------------- CONTINUOUS ANIMATION ----------------

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 240

if arm.animation_data:
    arm.animation_data_clear()

step = 30   # new pose every 30 frames
scene.frame_set(1)
keyframe_all()

# Generating continuous motion
for frame in range(step, 241, step):

    scene.frame_set(frame)

    new_pose = random_gesture()
    apply_gesture(new_pose)
    keyframe_all()

scene.frame_set(1)

print("Continuous random motion created!")