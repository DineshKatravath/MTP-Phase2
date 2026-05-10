import config
import utils
import random
import bpy
import sys
import time

# ---------- CORE HELPERS ----------

def curl_finger(arm, finger_name, amount, strength=1.0):
    bones = config.FINGERS[finger_name]

    for i, bone in enumerate(bones):
        weight = config.CURL_WEIGHTS[min(i, len(config.CURL_WEIGHTS)-1)]
        utils.rotate_bone(
            arm,
            bone,
            config.CURL_AXIS,
            amount * weight * strength
        )


def apply_bias(arm):
    for finger, val in config.BIAS.items():
        base_bone = config.FINGERS[finger][0]
        utils.rotate_bone(arm, base_bone, config.AXIS_Z, val)

def micro_noise(arm, amount=2):
    for pb in arm.pose.bones:
        r = list(pb.rotation_euler)
        r[0] += utils.deg_to_rad(random.uniform(-amount, amount))
        r[1] += utils.deg_to_rad(random.uniform(-amount, amount))
        r[2] += utils.deg_to_rad(random.uniform(-amount, amount))
        pb.rotation_euler = r


# ---------- GESTURE LIBRARY ----------

def open_hand():
    arm = utils.get_armature()
    utils.reset_hand(arm)
    apply_bias(arm)


def fist(strength=1.0, noise=0):
    arm = utils.get_armature()
    utils.reset_hand(arm)
    apply_bias(arm)

    for f in ["index","middle","ring","pinky"]:
        curl_finger(arm, f, 75, strength)

    curl_finger(arm, "thumb", 60, strength)

    if noise > 0:
        micro_noise(arm, noise)

def point():
    arm = utils.get_armature()
    utils.reset_hand(arm)
    apply_bias(arm)

    curl_finger(arm, "middle", 65)
    curl_finger(arm, "ring", 65)
    curl_finger(arm, "pinky", 65)
    curl_finger(arm, "thumb", 35)


def peace():
    arm = utils.get_armature()
    utils.reset_hand(arm)
    apply_bias(arm)

    curl_finger(arm, "ring", 65)
    curl_finger(arm, "pinky", 65)
    curl_finger(arm, "thumb", 30)


def pinch():
    arm = utils.get_armature()
    utils.reset_hand(arm)
    apply_bias(arm)

    curl_finger(arm, "index", 40)
    curl_finger(arm, "thumb", 40)
    

# Automatic dataset generation

def generate_dataset(samples=50):
    import random
    arm = utils.get_armature()
    scene = bpy.context.scene

    for i in range(samples):
        scene.frame_set(i)

        s = random.uniform(0.2, 1.0)
        n = random.uniform(0, 2)

        fist(strength=s, noise=n)

        for pb in arm.pose.bones:
            pb.keyframe_insert(data_path="rotation_euler")

