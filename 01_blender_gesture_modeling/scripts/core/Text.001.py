import bpy
import math

# Active armature
arm = bpy.context.object

FINGERS = {
    "index":  ["index_01", "index_02", "index_03"],
    "middle": ["middle_01", "middle_02", "middle_03"],
    "ring":   ["ring_01", "ring_02", "ring_03"],
    "pinky":  ["pinky_01", "pinky_02", "pinky_03"],
    "thumb":  ["thumb_01", "thumb_02"] 
}

print("Finger groups ready")


# Utility Functions

def clamp(val, minv, maxv):
    return max(minv, min(maxv, val))

def rotate_bone(bone_name, axis, angle_deg):
    pb = arm.pose.bones.get(bone_name)
    if not pb:
        return
    pb.rotation_mode = 'XYZ'
    angle = math.radians(angle_deg)
    rot = [0,0,0]
    rot[axis] = angle
    pb.rotation_euler.x += angle

def curl_finger(name, amount_deg):
    weights = [1.0, 0.7, 0.4]
    bones = FINGERS[name]

    for i, b in enumerate(bones):
        rotate_bone(b, 0, amount_deg * weights[i])

def spread_finger(name, amount_deg):
    # axis 2 = Z spread
    bones = FINGERS[name]
    # spread mainly happens at base bone
    rotate_bone(bones[0], 2, amount_deg)


def reset_hand():
    for pb in arm.pose.bones:
        pb.rotation_euler = (0,0,0)


# Adding tiny bias so that gestures dont look robotic
BIAS = {
    "index":  -2,
    "middle": 0,
    "ring":   3,
    "pinky":  6
}

def apply_bias():
    for f, val in BIAS.items():
        bones = FINGERS[f]
        rotate_bone(bones[0], 2, val)


# Some Actions

def open_hand():
    reset_hand()
    apply_bias()

def fist():
    reset_hand()
    apply_bias()
    for f in ["index","middle","ring","pinky"]:
        curl_finger(f, 80)
    curl_finger("thumb", 80)

def point():
    reset_hand()
    apply_bias()
    curl_finger("middle", 65)
    curl_finger("ring", 65)
    curl_finger("pinky", 65)
    curl_finger("thumb", 30)

def pinch():
    reset_hand()
    apply_bias()
    curl_finger("index", 40)
    curl_finger("thumb", 35)

def grab():
    reset_hand()
    apply_bias()
    for f in ["index","middle","ring","pinky"]:
        curl_finger(f, 50)
    curl_finger("thumb", 30)


fist()
#point()
#pinch()
#grab()
#open_hand()


# KeyFrame Utility 
def keyframe_all(frame):
    bpy.context.scene.frame_set(frame)
    for pb in arm.pose.bones:
        pb.rotation_mode = 'XYZ' 
        pb.keyframe_insert(data_path="rotation_euler")
        
# Animate Curl Over Time
def animate_curl(finger, start, end, max_angle):
    for f in range(start, end+1):
        t = (f-start)/(end-start)
        angle = max_angle * t

        reset_hand()
        apply_bias()
        curl_finger(finger, angle)

        keyframe_all(f)

#animate_curl("index", 1, 20, 60)

def animate_fist(start, end):
    reset_hand()
    apply_bias()

    for f in range(start, end+1):
        bpy.context.scene.frame_set(f)

        t = (f-start)/(end-start)

        for finger in ["index","middle","ring","pinky"]:
            curl_finger(finger, 65*t)

        curl_finger("thumb", 40*t)

        keyframe_all(f)

        
#animate_fist(1, 30)
