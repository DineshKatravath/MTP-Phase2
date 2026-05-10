import bpy
import json
import math
from mathutils import Vector

# ---------------- CONFIG ----------------

ARMATURE_NAME = "Armature"
FILEPATH = bpy.path.abspath("//hand_keypoints_6dof.json")

arm = bpy.data.objects[ARMATURE_NAME]
pb = arm.pose.bones

# ---------------- LOAD DATA ----------------

with open(FILEPATH, "r") as f:
    data = json.load(f)

# ---------------- HELPERS ----------------

def set_bone_direction(bone, head, tail):
    """
    Align bone to point from head → tail
    """
    direction = (Vector(tail) - Vector(head)).normalized()

    bone.rotation_mode = 'QUATERNION'
    bone.rotation_quaternion = direction.to_track_quat('Y', 'Z')


# ---------------- APPLY FRAME ----------------

def angle(v1, v2):
    v1 = v1.normalized()
    v2 = v2.normalized()
    return math.acos(max(-1, min(1, v1.dot(v2))))

def apply_frame(frame_data):

    kp = frame_data["keypoints_global"]

    # -------- GLOBAL POSITION --------
    if "wrist" in kp:
        arm.location = Vector(kp["wrist"])

    # -------- FINGER PROCESS --------

    finger_map = {
        "index":  ["palm_index","index_01","index_02","index_03"],
        "middle": ["palm_middle","middle_01","middle_02","middle_03"],
        "ring":   ["palm_ring","ring_01","ring_02","ring_03"],
        "pinky":  ["palm_pinky","pinky_01","pinky_02","pinky_03"],
    }

    bone_map = {
        "index":  ["index_01","index_02","index_03"],
        "middle": ["middle_01","middle_02","middle_03"],
        "ring":   ["ring_01","ring_02","ring_03"],
        "pinky":  ["pinky_01","pinky_02","pinky_03"],
    }

    for name, joints in finger_map.items():

        if any(j not in kp for j in joints):
            continue

        p0 = Vector(kp[joints[0]])
        p1 = Vector(kp[joints[1]])
        p2 = Vector(kp[joints[2]])
        p3 = Vector(kp[joints[3]])

        # vectors
        v1 = p1 - p0
        v2 = p2 - p1
        v3 = p3 - p2

        # angles
        a1 = angle(v1, v2)
        a2 = angle(v2, v3)

        bones = bone_map[name]

        # MCP (base)
        if bones[0] in pb:
            b = pb[bones[0]]
            b.rotation_mode = 'XYZ'
            b.rotation_euler[0] = a1 * 1.5

        # PIP
        if bones[1] in pb:
            b = pb[bones[1]]
            b.rotation_mode = 'XYZ'
            b.rotation_euler[0] = a1

        # DIP
        if bones[2] in pb:
            b = pb[bones[2]]
            b.rotation_mode = 'XYZ'
            b.rotation_euler[0] = a2

    # -------- THUMB --------

    thumb_map = ["palm_thumb","thumb_01","thumb_02"]

    if all(k in kp for k in thumb_map):

        p0 = Vector(kp["palm_thumb"])
        p1 = Vector(kp["thumb_01"])
        p2 = Vector(kp["thumb_02"])

        v1 = p1 - p0
        v2 = p2 - p1

        a = angle(v1, v2)

        if "thumb_01" in pb:
            pb["thumb_01"].rotation_mode = 'XYZ'
            pb["thumb_01"].rotation_euler[0] = a * 1.2

        if "thumb_02" in pb:
            pb["thumb_02"].rotation_mode = 'XYZ'
            pb["thumb_02"].rotation_euler[0] = a

# ---------------- ANIMATION ----------------

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = len(data)

for i, frame_data in enumerate(data):

    frame = i + 1
    scene.frame_set(frame)

    apply_frame(frame_data)

    # -------- KEYFRAME --------
    arm.keyframe_insert(data_path="location")

    for bone in pb:
        if bone.rotation_mode == 'QUATERNION':
            bone.keyframe_insert(data_path="rotation_quaternion")

print("Reconstruction complete")