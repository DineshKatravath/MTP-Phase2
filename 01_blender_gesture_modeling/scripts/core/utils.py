import bpy
import math

def get_armature():
    obj = bpy.context.object
    if obj and obj.type == 'ARMATURE':
        return obj

    # fallback: first armature in scene
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            return o
    return None


def deg_to_rad(d):
    return math.radians(d)


def reset_hand(arm):
    bpy.ops.object.mode_set(mode='POSE')
    for pb in arm.pose.bones:
        pb.rotation_euler = (0, 0, 0)


def rotate_bone(arm, bone_name, axis, degrees):
    pb = arm.pose.bones.get(bone_name)
    if not pb:
        return
    pb.rotation_mode = 'XYZ'
    r = list(pb.rotation_euler)
    r[axis] = deg_to_rad(degrees)
    pb.rotation_euler = r
