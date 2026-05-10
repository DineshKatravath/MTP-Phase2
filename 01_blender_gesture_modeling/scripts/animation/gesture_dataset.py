import bpy
import math
import json 
import random
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
    "ring": 4,
    "pinky": 6
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

            noise = random.uniform(-2, 2)
            b.rotation_euler[AXIS_X] = deg(curl * weight + noise)

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

    # WRIST (with smooth motion)
    if "wrist" in pb:

        w = pb["wrist"]
        w.rotation_mode='QUATERNION'

        global prev_wrist_offset

        # initialize if first time
        if "prev_wrist_offset" not in globals():
            prev_wrist_offset = [0,0,0]

        # smooth random motion (random walk)
#        prev_wrist_offset = [
#            prev_wrist_offset[0] + random.uniform(-360,360),
#            prev_wrist_offset[1] + random.uniform(-360,360),
#            prev_wrist_offset[2] + random.uniform(-360,360),
#        ]

        # clamp (important!)
#        prev_wrist_offset = [
#            max(-30, min(30, v)) for v in prev_wrist_offset
#        ]

        e = Euler((
            deg(wrist[0] + prev_wrist_offset[0]),
            deg(wrist[1] + prev_wrist_offset[1]),
            deg(wrist[2] + prev_wrist_offset[2])
        ), 'XYZ')

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
    "fingers": {"index": 2,"middle": 8, "ring": 100,"pinky": 100},
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


# All Available gestures

GESTURES = [
    OPEN, FIST, THUMBS_UP, THUMBS_DOWN,
    WAVE_LEFT, WAVE_RIGHT, V_SIGN, POINT, PINCH, ROCK
]

GESTURE_NAMES = [
    "OPEN","FIST","THUMBS_UP","THUMBS_DOWN",
    "WAVE_LEFT","WAVE_RIGHT","V_SIGN","POINT","PINCH","ROCK"
]


# ---------------- TIMELINE GENERATION ----------------

TOTAL_FRAMES = 10000
HOLD_STEP = 70
TRANSITION_STEP = 30

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = TOTAL_FRAMES

timeline = []

frame = 1
current_pos = [0,0,0]

# to genrate end position for a particular gesture within the limits of the bounded box
BOX = {"x":(-40,40),"y":(-40,40),"z":(-40,40)}
STEP_LIMIT = 15

def bounded_target(current_pos):
#    target = [
#        current_pos[0] + random.uniform(-STEP_LIMIT, STEP_LIMIT),
#        current_pos[1] + random.uniform(-STEP_LIMIT, STEP_LIMIT),
#        current_pos[2] + random.uniform(-STEP_LIMIT, STEP_LIMIT),
#    ]
    target = [
        current_pos[0] ,
        current_pos[1] ,
        current_pos[2] ,
    ]
    return [
        max(BOX["x"][0], min(BOX["x"][1], target[0])),
        max(BOX["y"][0], min(BOX["y"][1], target[1])),
        max(BOX["z"][0], min(BOX["z"][1], target[2]))
    ]

while frame < TOTAL_FRAMES:

    idx = random.randint(0,len(GESTURES)-1)
    gesture = GESTURES[idx]

    start_frame = frame
    start_pos = current_pos.copy()

    end_pos = bounded_target(current_pos)

    end_frame = frame + HOLD_STEP - 1

    timeline.append((start_frame, gesture, tuple(start_pos)))
    timeline.append((end_frame, gesture, tuple(end_pos)))

    current_pos = end_pos
    frame += (HOLD_STEP + TRANSITION_STEP)

timeline = [t for t in timeline if t[0] <= TOTAL_FRAMES]

# ---------------- APPLY TIMELINE ----------------

for frame, gesture, location in timeline:
    scene.frame_set(frame)
    apply_gesture(gesture)
    arm.location = location
    keyframe_all()

# ---------------- LABEL GENERATION ----------------

frame_labels = {}

for i in range(0,len(timeline),2):

    start_frame, gesture, _ = timeline[i]
    end_frame, _, _ = timeline[i+1]

    name = GESTURE_NAMES[GESTURES.index(gesture)]

    for f in range(start_frame, end_frame+1):
        frame_labels[f] = name

    if i+2 < len(timeline):
        next_start = timeline[i+2][0]
        for f in range(end_frame+1, next_start):
            frame_labels[f] = "transition"

# ---------------- DATA EXTRACTION ----------------

def extract_frame_data(frame):

    scene.frame_set(frame)

    data = {
        "global":{
            "position":list(arm.location),
            "rotation_quaternion":list(arm.rotation_quaternion)
        }
    }

    joints = {}

    for bone in pb:
        world_pos = arm.matrix_world @ bone.matrix @ bone.head

        if bone.rotation_mode=='QUATERNION':
            rot_q = list(bone.rotation_quaternion)
            rot_e = list(bone.rotation_quaternion.to_euler('XYZ'))
        else:
            rot_e = list(bone.rotation_euler)
            rot_q = list(bone.rotation_euler.to_quaternion())

        joints[bone.name] = {
            "world_position":[world_pos.x,world_pos.y,world_pos.z],
            "rotation_euler":rot_e,
            "rotation_quaternion":rot_q
        }

    data["joints"] = joints
    return data

dataset = []
prev = None

for f in range(scene.frame_start, scene.frame_end+1):

    frame_data = extract_frame_data(f)

    if prev:
        vel = [
            frame_data["global"]["position"][i] - prev["global"]["position"][i]
            for i in range(3)
        ]
    else:
        vel = [0,0,0]

    frame_data["global_velocity"] = vel

    if prev:
        for j in frame_data["joints"]:
            curr = frame_data["joints"][j]["world_position"]
            prev_j = prev["joints"][j]["world_position"]
            frame_data["joints"][j]["velocity"] = [curr[i]-prev_j[i] for i in range(3)]
    else:
        for j in frame_data["joints"]:
            frame_data["joints"][j]["velocity"] = [0,0,0]

    dataset.append({
        "frame": f,
        "gesture": frame_labels.get(f,"transition"),
        **frame_data
    })

    prev = frame_data

# ---------------- SAVE ----------------

filepath = bpy.path.abspath("//hand_no_motion_dataset.json")

with open(filepath,"w") as file:
    json.dump(dataset,file,indent=2)

print(f"Saved dataset to {filepath}")