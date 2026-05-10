import bpy
import socket
import json
import threading
import math
import numpy as np
import mathutils
from mathutils import Vector

# ---------------- CONFIG ----------------

ARMATURE_NAME = "Armature"
HOST = "127.0.0.1"
PORT = 9999

CURL_MIN = 0.05
CURL_MAX = 0.45
DEBUG = False

arm = bpy.data.objects[ARMATURE_NAME]
pb = arm.pose.bones

latest_data = None
prev_q = None
prev_normal = None

# ---------------- HELPERS ----------------

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

def finger_curl(p1, p2, p3, palm_normal):
    v1 = p2 - p1
    v2 = p3 - p2
    n = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
    v1 = v1 - np.dot(v1, n) * n
    v2 = v2 - np.dot(v2, n) * n
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    cos_angle = np.dot(v1, v2) / (len1 * len2)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    curl = angle / math.pi
    return np.clip(curl, 0, 1)

# ---------------- INIT BONE MODES (run once) ----------------

def init_bones():
    finger_map = {
        "index":  ["index_01",  "index_02",  "index_03" ],
        "middle": ["middle_01", "middle_02", "middle_03"],
        "ring":   ["ring_01",   "ring_02",   "ring_03"  ],
        "pinky":  ["pinky_01",  "pinky_02",  "pinky_03" ],
    }
    for joints in finger_map.values():
        for j in joints:
            pb[j].rotation_mode = 'XYZ'
    pb["thumb_01"].rotation_mode = 'XYZ'
    pb["thumb_02"].rotation_mode = 'XYZ'
    arm.rotation_mode = 'QUATERNION'

init_bones()

# ---------------- APPLY FRAME ----------------

def apply_frame(kp):
    global prev_q, prev_normal

    # -------- WRIST + GLOBAL TRANSFORM --------
    # palm_normal_np defaults to straight-on if wrist missing
    palm_normal_np = np.array([0.0, 0.0, 1.0])

    if all(k in kp for k in ["wrist", "palm_index", "palm_pinky", "thumb_02"]):

        w = Vector(kp["wrist"])
        i = Vector(kp["palm_index"])
        p = Vector(kp["palm_pinky"])
        t = Vector(kp["thumb_02"])

        # -------- POSITION --------
        loc = Vector((
            w.x * 80,
            (0.5 - w.y) * 80,
            (w.z - 0.5) * 80
        ))
        arm.matrix_world.translation = loc

        # -------- BASE VECTORS --------
        forward = ((i + p) * 0.5 - w).normalized()
        right = (i - p).normalized()
        palm_normal = -(forward.cross(right).normalized())
        right = forward.cross(palm_normal).normalized()

        palm_normal_np = np.array([palm_normal.x, palm_normal.y, palm_normal.z])

        rot_matrix = mathutils.Matrix((
            (right.x,       right.y,       right.z      ),
            (forward.x,     forward.y,     forward.z    ),
            (palm_normal.x, palm_normal.y, palm_normal.z),
        )).transposed()

        q = rot_matrix.to_quaternion()

        correction = mathutils.Euler((
            math.radians(-90),
            math.radians(90),
            math.radians(180)
        )).to_quaternion()

        q = q @ correction

        if prev_q is not None:
            if prev_q.dot(q) < 0:
                q = -q
            q = prev_q.slerp(q, 0.3)

        prev_q = q.copy()
        arm.rotation_quaternion = q

    # -------- FINGERS (independent of wrist block) --------

    finger_map = {
        "index":  ["index_01",  "index_02",  "index_03" ],
        "middle": ["middle_01", "middle_02", "middle_03"],
        "ring":   ["ring_01",   "ring_02",   "ring_03"  ],
        "pinky":  ["pinky_01",  "pinky_02",  "pinky_03" ],
    }

    # individual strength per finger — index higher to ensure full close
    strength_map = {
        "index":  6.0,
        "middle": 5.0,
        "ring":   4.5,
        "pinky":  4.0,
    }

    # individual curl max per finger — index needs lower max to close fully
    curl_max_map = {
        "index":  0.38,
        "middle": 0.45,
        "ring":   0.45,
        "pinky":  0.45,
    }

    for name, joints in finger_map.items():

        if not all(j in kp for j in joints):
            continue

        p1 = np.array(kp[joints[0]])
        p2 = np.array(kp[joints[1]])
        p3 = np.array(kp[joints[2]])

        curl = finger_curl(p1, p2, p3, palm_normal_np)

        if DEBUG:
            print(f"{name}: raw curl = {curl:.3f}")

        curl = (curl - CURL_MIN) / (curl_max_map[name] - CURL_MIN)
        curl = np.clip(curl, 0, 1)

        s = strength_map[name]

        pb[joints[0]].rotation_euler[0] = curl * s
        pb[joints[1]].rotation_euler[0] = curl * s * 0.9
        pb[joints[2]].rotation_euler[0] = curl * s * 0.7

    # -------- THUMB (fully independent) --------

    thumb_chain = ["palm_thumb", "thumb_01", "thumb_02"]

    if all(k in kp for k in thumb_chain):

        p0 = np.array(kp["palm_thumb"])
        p1 = np.array(kp["thumb_01"])
        p2 = np.array(kp["thumb_02"])

        curl = finger_curl(p0, p1, p2, palm_normal_np)

        if DEBUG:
            print(f"thumb: raw curl = {curl:.3f}")

        curl = (curl - CURL_MIN) / (0.45 - CURL_MIN)
        curl = np.clip(curl, 0, 1)

        pb["thumb_01"].rotation_euler[0] = -curl * 5.0
        pb["thumb_02"].rotation_euler[0] = -curl * 4.0


# ---------------- SOCKET ----------------

def listen():
    global latest_data

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)

    print("Waiting for MediaPipe...")
    conn, _ = s.accept()
    print("Connected!")

    buffer = ""

    while True:
        data = conn.recv(4096).decode()
        if not data:
            break
        buffer += data
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            latest_data = json.loads(line)


# ---------------- UPDATE LOOP ----------------

def update():
    if latest_data:
        apply_frame(latest_data)
    return 0.01


# ---------------- START ----------------

threading.Thread(target=listen, daemon=True).start()
bpy.app.timers.register(update)

print("Hand tracking started — fingers + orientation")