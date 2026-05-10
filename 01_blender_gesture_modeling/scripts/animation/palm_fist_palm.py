import bpy
import math

# ---------------- CONFIG ----------------

AXIS_X = 0  # Curl
AXIS_Y = 1  # Twist
AXIS_Z = 2  # Spread

FINGERS = {
    "index":  ["index_01", "index_02", "index_03"],
    "middle": ["middle_01", "middle_02", "middle_03"],
    "ring":   ["ring_01", "ring_02", "ring_03"],
    "pinky":  ["pinky_01", "pinky_02", "pinky_03"],
    "thumb":  ["thumb_01", "thumb_02"]
}

CURL_WEIGHTS = [1.0, 0.7, 0.4]

BIAS = {
    "index": -4,
    "middle": -2,
    "ring": 2,
    "pinky": 4
}

ARMATURE_NAME = "Armature"

# ---------------- SETUP ----------------

def deg(a):
    return math.radians(a)

arm = bpy.data.objects[ARMATURE_NAME]
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones

#  CLEAR OLD ANIMATION (IMPORTANT)
if arm.animation_data:
    arm.animation_data_clear()

# ---------------- POSE FUNCTION ----------------

def apply_pose(global_curl, global_spread):

    FINGER_OFFSET = {
        "index": -5,
        "middle": 0,
        "ring": 5,
        "pinky": 10
    }

    # ---- FINGERS ----
    for finger, bones in FINGERS.items():

        if finger == "thumb":
            continue

        finger_curl = global_curl + FINGER_OFFSET.get(finger, 0)
        finger_spread = global_spread * (1 - finger_curl / 100)

        for i, bone_name in enumerate(bones):
            if bone_name not in pb:
                continue

            b = pb[bone_name]
            b.rotation_mode = 'XYZ'
            weight = CURL_WEIGHTS[min(i, len(CURL_WEIGHTS)-1)]

            # Curl (all bones)
            b.rotation_euler[AXIS_X] = deg(finger_curl * weight)

            # Spread only base bone
            if i == 0:
                b.rotation_euler[AXIS_Z] = deg(finger_spread + BIAS.get(finger, 0))

    # ---- THUMB ----
    if "thumb_01" in pb:
        t1 = pb["thumb_01"]
        t1.rotation_mode = 'XYZ'
        t1.rotation_euler[AXIS_X] = deg(global_curl * 0.8)
        t1.rotation_euler[AXIS_Y] = deg(10)
        t1.rotation_euler[AXIS_Z] = deg(-15)

    if "thumb_02" in pb:
        t2 = pb["thumb_02"]
        t2.rotation_mode = 'XYZ'
        t2.rotation_euler[AXIS_X] = deg(global_curl * 0.6)


# ---------------- KEYFRAME FUNCTION ----------------

def keyframe_pose():
    for finger, bones in FINGERS.items():
        for bone_name in bones:
            if bone_name in pb:
                pb[bone_name].keyframe_insert(data_path="rotation_euler")

    if "wrist" in pb:
        pb["wrist"].keyframe_insert(data_path="rotation_euler")


# ---------------- ANIMATION ----------------

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 40

# Frame 1 → OPEN
scene.frame_set(1)
apply_pose(global_curl=0, global_spread=8)
keyframe_pose()

# Frame 20 → CLOSED FIST
scene.frame_set(20)
apply_pose(global_curl=90, global_spread=0)
keyframe_pose()

# Frame 40 → OPEN AGAIN
scene.frame_set(40)
apply_pose(global_curl=0, global_spread=8)
keyframe_pose()

scene.frame_set(1)

print("Animation Created Successfully!")