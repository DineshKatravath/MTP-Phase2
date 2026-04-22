# 08 — Live Hand Tracking

## What This Is

A real-time pipeline that captures hand movements from a webcam using MediaPipe, streams the 21 joint coordinates to Blender over a TCP socket, and drives the Blender rig live.

This serves two purposes: validating that the parameterization is correct (the Blender hand should mimic real hand movements), and as a demo of the full system.

---

## Scripts

| Script | Description |
|--------|-------------|
| `reciever.py` | v1 — basic curl from joint distances, global orientation |
| `reciever_v2.py` | v2 — improved curl scaling and strength per finger |
| `reciever_v3.py` | v3 — current best: plane-projected curl, flip-stable quaternion |

**Use `reciever_v3.py` for best results.**

---

## System Architecture

```
[Webcam]
   │
   ▼
[Sender script — runs in terminal]
  MediaPipe detects 21 hand landmarks
  Normalizes to wrist-relative coordinates
  Sends JSON over TCP socket on port 9999
   │
   ▼ (localhost:9999)
[Receiver script — runs in Blender]
  Receives JSON each frame
  Computes wrist quaternion from palm vectors
  Computes finger curl from projected angles
  Applies to rig every 10ms (100 Hz update)
```

---

## Sender Script (runs outside Blender)

The sender (from the earlier conversation) uses MediaPipe to detect the hand and streams joint positions:

```python
import cv2, mediapipe as mp, socket, json, numpy as np

# Sends per-frame:
{
    "wrist":       [x, y, z],
    "palm_index":  [x, y, z],
    "palm_middle": [x, y, z],
    "palm_ring":   [x, y, z],
    "palm_pinky":  [x, y, z],
    "palm_thumb":  [x, y, z],
    "index_01":    [x, y, z],
    ... (all 20 joint names matching Blender bones)
}
```

Install dependencies:
```bash
pip install mediapipe opencv-python
```

---

## Receiver — Version Comparison

### v1 (`reciever.py`)
- **Curl formula:** `1 - (tip-to-base distance) / (sum of bone lengths)`
- **Problem:** Distance-based curl is affected by wrist rotation — fingers appear to curl/uncurl when the wrist tilts even if they don't move

### v2 (`reciever_v2.py`)
- Same curl formula but added global hand orientation using wrist quaternion from palm vectors
- **Improvement:** Hand rotates correctly in 3D
- **Remaining issue:** Curl still affected by wrist tilt at extreme angles

### v3 (`reciever_v3.py`) — Current
- **Curl formula:** Project finger bone vectors onto the palm plane first, then compute the angle between projected vectors
- **Improvement:** Curl is now pure finger bending — completely independent of wrist orientation
- **Added:** Per-finger strength and curl_max calibration (index finger needs tighter calibration to close fully)
- **Added:** Quaternion dot-product flip detection before slerp (prevents sudden 360° flips)
- **Added:** Debug mode (`DEBUG = True`) to print raw curl values per finger

---

## Global Orientation Computation (v3)

The wrist quaternion is computed from three vectors derived from palm landmarks:

```python
w = Vector(kp["wrist"])
i = Vector(kp["palm_index"])
p = Vector(kp["palm_pinky"])

forward     = ((i + p) * 0.5 - w).normalized()   # wrist → mid palm
right       = (i - p).normalized()               # pinky → index
palm_normal = -(forward.cross(right).normalized()) # outward normal (negated for front-facing)
right       = forward.cross(palm_normal).normalized()  # reorthogonalized

rot_matrix  = Matrix([right, forward, palm_normal]).transposed()
q           = rot_matrix.to_quaternion() @ correction_quaternion
```

A correction quaternion (`Euler(-90°, 90°, 180°)`) is applied to align the MediaPipe coordinate system with Blender's.

---

## Finger Curl Computation (v3)

```python
def finger_curl(p1, p2, p3, palm_normal):
    v1 = p2 - p1   # first bone vector
    v2 = p3 - p2   # second bone vector
    # Project onto palm plane (remove component along palm normal)
    v1 = v1 - dot(v1, n) * n
    v2 = v2 - dot(v2, n) * n
    # Angle between projected vectors
    angle = arccos(dot(v1, v2) / (|v1| * |v2|))
    return angle / pi   # normalized to [0, 1]
```

---

## Calibration Parameters (v3)

```python
CURL_MIN = 0.05    # raw curl below this → 0 (fully open)
CURL_MAX = 0.45    # per-finger, see curl_max_map

curl_max_map = {
    "index":  0.38,   # index closes with less curl angle than others
    "middle": 0.45,
    "ring":   0.45,
    "pinky":  0.45,
}

strength_map = {
    "index":  6.0,
    "middle": 5.0,
    "ring":   4.5,
    "pinky":  4.0,
}
```

If fingers don't close fully or over-rotate, adjust `curl_max_map` and `strength_map`.

---

## How to Run

**Terminal 1 — Sender:**
```bash
python sender.py   # the MediaPipe script
```

**Blender — Receiver:**
1. Open `.blend` file with hand rig
2. Go to Scripting workspace
3. Open `reciever_v3.py`
4. Click Run Script
5. Blender console should print: `Hand tracking started — fingers + orientation`

Once the sender connects, the rig will start moving in real time.

---

## Known Issues / Current Limitations

- **Mirroring:** MediaPipe sometimes flips left/right hand detection. The receiver doesn't currently handle this case.
- **ML bridge not complete:** The goal of training a MediaPipe→Blender model for cleaner motion is still in progress. The main blocker is that Blender renders aren't detected by MediaPipe due to synthetic appearance.
- **Smoothing:** The quaternion slerp factor is 0.3 (30% toward new value each frame). Increasing this makes motion more responsive but jerkier; decreasing makes it smoother but laggy.
