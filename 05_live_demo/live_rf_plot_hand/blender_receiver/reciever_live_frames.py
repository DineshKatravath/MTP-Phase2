import bpy
import socket
import json
import threading
import math
import os
import shutil
import struct
import numpy as np
import mathutils
from mathutils import Vector

# ---------------- CONFIG ----------------

ARMATURE_NAME   = "Armature"
MESH_NAME       = "Plane005.001"
HOST            = "127.0.0.1"
PORT            = 9999

CURL_MIN        = 0.05
CURL_MAX        = 0.45
DEBUG           = False

FINGER_SMOOTH   = 0.25

PLY_OUTPUT_DIR  = "/Users/dinesh/Documents/mtp/hand_models/live_rf_plot_hand/hand_frames_live"

arm = bpy.data.objects[ARMATURE_NAME]
pb  = arm.pose.bones

# Socket thread pushes parsed dicts into this list.
# Main thread pops from it — one pop = one PLY saved.
# A list used as a queue avoids any missed-frame or duplicate issue.
frame_queue  = []
queue_lock   = threading.Lock()

prev_q      = None
prev_curls  = {}
frame_index = 0

# ---- clean output dir on every script run ----
if os.path.exists(PLY_OUTPUT_DIR):
    shutil.rmtree(PLY_OUTPUT_DIR)
os.makedirs(PLY_OUTPUT_DIR)
print(f"Output dir cleaned: {PLY_OUTPUT_DIR}")

# ---------------- HELPERS ----------------

def finger_curl(p1, p2, p3, palm_normal):
    v1 = p2 - p1
    v2 = p3 - p2
    n  = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
    v1 = v1 - np.dot(v1, n) * n
    v2 = v2 - np.dot(v2, n) * n
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1, 1)
    return float(np.clip(np.arccos(cos_angle) / math.pi, 0, 1))

# ---------------- INIT BONE MODES ----------------

def init_bones():
    finger_map = {
        "index":  ["index_01",  "index_02",  "index_03"],
        "middle": ["middle_01", "middle_02", "middle_03"],
        "ring":   ["ring_01",   "ring_02",   "ring_03"],
        "pinky":  ["pinky_01",  "pinky_02",  "pinky_03"],
    }
    for joints in finger_map.values():
        for j in joints:
            pb[j].rotation_mode = 'XYZ'
    pb["thumb_01"].rotation_mode = 'XYZ'
    pb["thumb_02"].rotation_mode = 'XYZ'
    arm.rotation_mode = 'QUATERNION'

init_bones()

# ---------------- PLY EXPORT ----------------
# Blender axes:  X=right  Y=up    Z=forward  (from your screenshot)
# Sionna/Mitsuba: X=right  Y=forward  Z=up
# Remap on export: export_x=bl_x, export_y=bl_z, export_z=bl_y

def save_ply_frame(mesh_obj, filepath):
    import bmesh

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj  = mesh_obj.evaluated_get(depsgraph)
    mesh      = eval_obj.to_mesh()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    verts = [v.co for v in mesh.vertices]
    faces = [p.vertices[:] for p in mesh.polygons]

    tmp_path = filepath + ".tmp"
    with open(tmp_path, 'wb') as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(verts)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {len(faces)}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        f.write(header.encode('ascii'))

        for v in verts:
            # Blender (X, Y, Z) → Sionna (X, Z, Y)
            f.write(struct.pack('<fff', v.x, v.z, v.y))

        for face in faces:
            f.write(struct.pack('<B', 3))
            f.write(struct.pack('<iii', face[0], face[1], face[2]))

    os.replace(tmp_path, filepath)  # atomic
    eval_obj.to_mesh_clear()

# ---------------- APPLY FRAME ----------------

def apply_frame(kp):
    global prev_q, prev_curls, frame_index

    palm_normal_np = np.array([0.0, 0.0, 1.0])

    if all(k in kp for k in ["wrist", "palm_index", "palm_pinky", "thumb_02"]):

        w = Vector(kp["wrist"])
        i = Vector(kp["palm_index"])
        p = Vector(kp["palm_pinky"])

        loc = Vector((w.x * 80, (0.5 - w.y) * 80, (w.z - 0.5) * 80))
        arm.matrix_world.translation = loc

        forward     = ((i + p) * 0.5 - w).normalized()
        right       = (i - p).normalized()
        palm_normal = forward.cross(right).normalized()
        right       = forward.cross(palm_normal).normalized()

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

    # -------- FINGERS --------

    finger_map = {
        "index":  ["index_01",  "index_02",  "index_03"],
        "middle": ["middle_01", "middle_02", "middle_03"],
        "ring":   ["ring_01",   "ring_02",   "ring_03"],
        "pinky":  ["pinky_01",  "pinky_02",  "pinky_03"],
    }
    strength_map = {"index": 6.0, "middle": 5.0, "ring": 4.5, "pinky": 4.0}
    curl_max_map = {"index": 0.38, "middle": 0.45, "ring": 0.45, "pinky": 0.45}

    for name, joints in finger_map.items():
        if not all(j in kp for j in joints):
            continue
        curl = finger_curl(
            np.array(kp[joints[0]]),
            np.array(kp[joints[1]]),
            np.array(kp[joints[2]]),
            palm_normal_np
        )
        curl = float(np.clip((curl - CURL_MIN) / (curl_max_map[name] - CURL_MIN), 0, 1))
        if name in prev_curls:
            curl = prev_curls[name] + (curl - prev_curls[name]) * FINGER_SMOOTH
        prev_curls[name] = curl

        s = strength_map[name]
        pb[joints[0]].rotation_euler[0] = curl * s
        pb[joints[1]].rotation_euler[0] = curl * s * 0.9
        pb[joints[2]].rotation_euler[0] = curl * s * 0.7

    # -------- THUMB --------

    if all(k in kp for k in ["palm_thumb", "thumb_01", "thumb_02"]):
        curl = finger_curl(
            np.array(kp["palm_thumb"]),
            np.array(kp["thumb_01"]),
            np.array(kp["thumb_02"]),
            palm_normal_np
        )
        curl = float(np.clip((curl - CURL_MIN) / (0.45 - CURL_MIN), 0, 1))
        if "thumb" in prev_curls:
            curl = prev_curls["thumb"] + (curl - prev_curls["thumb"]) * FINGER_SMOOTH
        prev_curls["thumb"] = curl
        pb["thumb_01"].rotation_euler[0] = -curl * 5.0
        pb["thumb_02"].rotation_euler[0] = -curl * 4.0

    # -------- SAVE PLY --------
    mesh_obj = bpy.data.objects.get(MESH_NAME)
    if mesh_obj:
        filepath = os.path.join(PLY_OUTPUT_DIR, f"hand_{frame_index:04d}.ply")
        save_ply_frame(mesh_obj, filepath)
        if DEBUG:
            print(f"Saved: {filepath}")
        frame_index += 1


# ---------------- SOCKET ----------------
# Every complete JSON line is a unique keypoint packet.
# Push each one onto frame_queue — main thread drains it one at a time.

def listen():
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
            with queue_lock:
                frame_queue.append(json.loads(line))


# ---------------- UPDATE LOOP ----------------
# Blender calls this every 10 ms on the main thread.
# Pop exactly ONE packet per tick — each unique packet → one unique PLY.
# If queue is empty, nothing happens (no duplicate saves).

def update():
    with queue_lock:
        if not frame_queue:
            return 0.01
        kp = frame_queue.pop(0)   # consume oldest packet

    apply_frame(kp)
    return 0.01


# ---------------- START ----------------

threading.Thread(target=listen, daemon=True).start()
bpy.app.timers.register(update)

print(f"Hand tracking started — saving PLY frames to: {PLY_OUTPUT_DIR}")