import bpy
import math
import json 
from mathutils import Euler

# CONFIG

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
    "index": -12,
    "middle": 4,
    "ring": 6,
    "pinky": 10
}

def deg(a):
    return math.radians(a)

# SETUP

arm = bpy.data.objects[ARMATURE_NAME]
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')

pb = arm.pose.bones

if arm.animation_data:
    arm.animation_data_clear()

# APPLY GESTURE FUNCTION

def apply_gesture(g):

    finger_curls = g["fingers"]
    spread = g["spread"]
    palm = g["palm"]
    wrist = g["wrist"]
    thumb = g["thumb"]

    # FINGERS 
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

    # THUMB 
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

    # PALM
    for p in PALMS:

        if p in pb:

            bone = pb[p]
            bone.rotation_mode='XYZ'
            bone.rotation_euler[AXIS_X] = deg(palm)

    # WRIST
    if "wrist" in pb:

        w = pb["wrist"]
        w.rotation_mode='QUATERNION'

        e = Euler((deg(wrist[0]),deg(wrist[1]),deg(wrist[2])),'XYZ')
        w.rotation_quaternion = e.to_quaternion()


# KEYFRAME
   
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
    "wrist": (-160,200,90)
}

FIST = {
    "fingers": {"index":85,"middle":90,"ring":95,"pinky":100},
    "thumb": 90,
    "spread": 10,
    "palm": 25,
    "wrist": (-160,200,90)
}

THUMBS_UP = {
    "fingers": {"index":90,"middle":95,"ring":95,"pinky":95},
    "thumb": -10,
    "spread": 0,
    "palm": 20,
    "wrist": (-160,180,180)
}

THUMBS_DOWN = {
    "fingers": {"index":90,"middle":95,"ring":95,"pinky":95},
    "thumb": -10,
    "spread": 0,
    "palm": 20,
    "wrist": (30,170,-330)
}

WAVE_LEFT = {
    "fingers": {"index":0,"middle":0,"ring":0,"pinky":0},
    "thumb": 0,
    "spread": 6,
    "palm": 0,
    "wrist": (-160,200,60)
}

WAVE_RIGHT = {
    "fingers": {"index":0,"middle":0,"ring":0,"pinky":0},
    "thumb": 0,
    "spread": 6,
    "palm": 0,
    "wrist": (-160,200,120)
}

V_SIGN = {
    "fingers": {"index": 2,"middle": 8, "ring": 85,"pinky": 95},
    "thumb": 85,
    "spread": 0,
    "palm": 10,
    "wrist": (-160, 200, 90)
}


POINT = {
    "fingers": {"index": 0,"middle": 95,"ring": 85,"pinky": 95},
    "thumb": 15,
    "spread": 10,
    "palm": 10,
    "wrist": (-160, 200, 90)
}

PINCH = {
    "fingers": {"index": 30,"middle": 90,"ring": 95,"pinky": 100},
    "thumb": 40,
    "spread": 0,
    "palm": 5,
    "wrist": (-160, 200, 90)
}

ROCK = {
    "fingers": {"index": 0,"middle": 100,"ring": 100,"pinky": 0},
    "thumb": 20,
    "spread": 20,
    "palm": 5,
    "wrist": (-160, 200, 90)
}

# ANIMATION

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 360

#timeline = [

#(1,OPEN,(0,0,0)),
#(30,FIST,(0,0,0)),
#(60,OPEN,(0,0,0)),

## wave
#(90,WAVE_RIGHT,(0,0,0)),
#(120,WAVE_LEFT,(0,0,0)),
#(150,WAVE_RIGHT,(0,0,0)),

## move forward
#(210,OPEN,(0,30,30)),

## move backward
#(270,OPEN,(0,-30,30)),

## thumbs
#(300,THUMBS_UP,(0,0,0)),
#(330,THUMBS_DOWN,(0,0,0)),

#(360,OPEN,(0,0,0))

#]

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 1000

timeline = [

(1,    OPEN, (0,0,0)),
(101,  FIST, (0,0,0)),
(201,  THUMBS_UP, (0,0,0)),
(301,  THUMBS_DOWN, (0,0,0)),
(401,  WAVE_RIGHT, (0,0,0)),
(501,  WAVE_LEFT, (0,0,0)),
(601,  V_SIGN, (0,0,0)),
(701,  POINT, (0,0,0)),
(801,  PINCH, (0,0,0)),
(901,  ROCK, (0,0,0)),

]

for frame,gesture,location in timeline:

    scene.frame_set(frame)

    apply_gesture(gesture)

    arm.location = location

    keyframe_all()

scene.frame_set(1)

print("Gesture animation created successfully!")

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