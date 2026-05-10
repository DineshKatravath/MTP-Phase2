import bpy
import json
from mathutils import Vector, Quaternion, Euler

# -------- CONFIG --------
ARMATURE_NAME = "Armature"
JSON_PATH = bpy.path.abspath("//hand_motion_full.json")

# -------- LOAD DATA --------
with open(JSON_PATH, "r") as f:
    dataset = json.load(f)

# -------- SETUP --------
arm = bpy.data.objects[ARMATURE_NAME]
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones

scene = bpy.context.scene

# Clear previous animation
if arm.animation_data:
    arm.animation_data_clear()

# -------- APPLY FRAME --------
def apply_frame(frame_data):

    # -------- GLOBAL TRANSFORM --------
    global_data = frame_data["global"]

    arm.location = Vector(global_data["position"])

    arm.rotation_mode = 'QUATERNION'
    arm.rotation_quaternion = Quaternion(global_data["rotation_quaternion"])

    # -------- BONES --------
    joints = frame_data["joints"]

    for bone_name, joint_data in joints.items():

        if bone_name not in pb:
            continue

        bone = pb[bone_name]

        # Prefer quaternion (stable)
        if "rotation_quaternion" in joint_data:
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = Quaternion(joint_data["rotation_quaternion"])

        elif "rotation_euler" in joint_data:
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = Euler(joint_data["rotation_euler"], 'XYZ')

    # -------- KEYFRAME --------

    # Object (global)
    arm.keyframe_insert(data_path="location")
    arm.keyframe_insert(data_path="rotation_quaternion")

    # Bones
    for bone in pb:
        if bone.rotation_mode == 'QUATERNION':
            bone.keyframe_insert(data_path="rotation_quaternion")
        else:
            bone.keyframe_insert(data_path="rotation_euler")


# -------- REBUILD ANIMATION --------

scene.frame_start = dataset[0]["frame"]
scene.frame_end = dataset[-1]["frame"]

for frame_data in dataset:

    f = frame_data["frame"]
    scene.frame_set(f)

    apply_frame(frame_data)

# Reset to start
scene.frame_set(scene.frame_start)

print(" Scene reconstructed perfectly from JSON!")