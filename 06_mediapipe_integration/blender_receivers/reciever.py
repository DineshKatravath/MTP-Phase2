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

arm = bpy.data.objects[ARMATURE_NAME]
pb = arm.pose.bones

latest_data = None
prev_q = None
prev_normal = None

# ---------------- HELPERS ----------------

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)

def finger_curl(p1, p2, p3):
    d1 = np.linalg.norm(p1 - p2)
    d2 = np.linalg.norm(p2 - p3)
    d3 = np.linalg.norm(p1 - p3)
    curl = 1 - (d3 / (d1 + d2 + 1e-6))
    return np.clip(curl, 0, 1)

def ortho_normalize(x, y):
    x = x.normalized()
    y = (y - x * x.dot(y)).normalized()
    z = x.cross(y).normalized()
    return x, y, z

# ---------------- APPLY FRAME ----------------

def apply_frame(kp):
    global prev_q

    # -------- WRIST + GLOBAL TRANSFORM --------
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
        # forward = wrist to mid knuckle
        forward = ((i + p) * 0.5 - w).normalized()

        # right = always from pinky toward index (consistent direction)
        right = (i - p).normalized()

        # palm_normal = negated to fix front/back swap
        palm_normal = -(forward.cross(right).normalized())

        # reorthogonalize right from the stable normal + forward
        right = forward.cross(palm_normal).normalized()

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

        # -------- SMOOTHING --------
        if prev_q is not None:
            q = prev_q.slerp(q, 0.6)

        prev_q = q.copy()

        arm.rotation_mode = 'QUATERNION'
        arm.rotation_quaternion = q

    # -------- FINGERS --------

    finger_map = {
        "index":  ["index_01",  "index_02",  "index_03" ],
        "middle": ["middle_01", "middle_02", "middle_03"],
        "ring":   ["ring_01",   "ring_02",   "ring_03"  ],
        "pinky":  ["pinky_01",  "pinky_02",  "pinky_03" ],
    }

    strength = 2.5

    for name, joints in finger_map.items():

        if not all(j in kp for j in joints):
            continue

        p1 = np.array(kp[joints[0]])
        p2 = np.array(kp[joints[1]])
        p3 = np.array(kp[joints[2]])

        curl = finger_curl(p1, p2, p3)

        scale_map = {
            "index":  1.2,
            "middle": 1.0,
            "ring":   0.9,
            "pinky":  0.8,
        }
        s = scale_map[name]

        pb[joints[0]].rotation_mode = 'XYZ'
        pb[joints[0]].rotation_euler[0] = curl * s * strength

        pb[joints[1]].rotation_mode = 'XYZ'
        pb[joints[1]].rotation_euler[0] = curl * s * strength * 0.8

        pb[joints[2]].rotation_mode = 'XYZ'
        pb[joints[2]].rotation_euler[0] = curl * s * strength * 0.6

    # -------- THUMB --------

    thumb_chain = ["palm_thumb", "thumb_01", "thumb_02"]

    if all(k in kp for k in thumb_chain):

        p0 = np.array(kp["palm_thumb"])
        p1 = np.array(kp["thumb_01"])
        p2 = np.array(kp["thumb_02"])

        curl = finger_curl(p0, p1, p2)

        pb["thumb_01"].rotation_mode = 'XYZ'
        pb["thumb_01"].rotation_euler[0] = curl * 2.0

        pb["thumb_02"].rotation_mode = 'XYZ'
        pb["thumb_02"].rotation_euler[0] = curl * 1.5


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