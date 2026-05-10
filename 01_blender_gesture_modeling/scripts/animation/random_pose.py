import bpy
import random
import math


# ---------- CONFIG ----------
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

CURL_AXIS = AXIS_X

FINGERS = {
    "index":  ["index_01", "index_02", "index_03"],
    "middle": ["middle_01", "middle_02", "middle_03"],
    "ring":   ["ring_01", "ring_02", "ring_03"],
    "pinky":  ["pinky_01", "pinky_02", "pinky_03"],
    "thumb":  ["thumb_01", "thumb_02"] 
}

# Curl distribution (base, mid, tip)
CURL_WEIGHTS = [1.0, 0.7, 0.4]

# Slight inward arc bias per finger
BIAS = {
    "index": -4,
    "middle": -2,
    "ring": 2,
    "pinky": 4
}



ARMATURE_NAME = "Armature"

def deg(a):
    return math.radians(a)

def rand(min_d, max_d):
    return deg(random.uniform(min_d, max_d))

arm = bpy.data.objects[ARMATURE_NAME]
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones

# ---------------- WRIST ----------------
if "wrist" in pb:
    w = pb["wrist"]
    w.rotation_mode = 'XYZ'
    w.rotation_euler[AXIS_X] = 0#rand(-180, 180)
    w.rotation_euler[AXIS_Y] = 0#rand(-180, 180)
    w.rotation_euler[AXIS_Z] = 0#rand(-180, 180)

# ---------------- PALMS ----------------
for name in ["palm_index","palm_middle","palm_ring","palm_pinky","palm_thumb"]:
    if name in pb:
        p = pb[name]
        p.rotation_mode = 'XYZ'
        p.rotation_euler[CURL_AXIS] = rand(-15, 25)

# ---------------- FINGERS ----------------
global_curl = random.uniform(10, 85)
global_spread = random.uniform(-8, 8)

FINGER_OFFSET = {
    "index": -5,
    "middle": 0,
    "ring": 5,
    "pinky": 10
}

for finger, bones in FINGERS.items():

    finger_curl = global_curl + FINGER_OFFSET.get(finger, 0)
    finger_spread = global_spread * (1 - finger_curl / 100)

    for i, bone_name in enumerate(bones):
        if bone_name not in pb:
            continue

        b = pb[bone_name]
        b.rotation_mode = 'XYZ'
        weight = CURL_WEIGHTS[min(i, len(CURL_WEIGHTS)-1)]

        # ---- CURL (all bones) ----
        b.rotation_euler[AXIS_X] = deg(finger_curl * weight)

        # ---- SPREAD ONLY BASE ----
        if i == 0:
            b.rotation_euler[AXIS_Z] = deg(finger_spread + BIAS.get(finger, 0))


# ---------------- THUMB EXTRA ROTATION ----------------
if "thumb_01" in pb:
    t1 = pb["thumb_01"]
    t1.rotation_mode = 'XYZ'
    t1.rotation_euler[AXIS_X] = rand(-30, 75)
    t1.rotation_euler[AXIS_Y] = rand(-20, 20)
    t1.rotation_euler[AXIS_Z] = rand(-30, 30)

print("Random pose generated using config!")

if "thumb_02" in pb:
    t2 = pb["thumb_02"]
    t2.rotation_mode = 'XYZ'
    t2.rotation_euler[AXIS_X] = rand(0, 50)
