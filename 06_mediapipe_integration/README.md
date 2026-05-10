# 06 — MediaPipe Integration

This module implements the real-time hand capture bridge: a MediaPipe sender extracts 21 hand landmark coordinates from a live webcam and streams them over TCP to a Blender receiver, which maps the landmarks to bone rotations and drives the hand armature in near real time.

## Background

While scripted animations produce controlled, reproducible datasets, they require every pose to be defined manually. The MediaPipe bridge allows arbitrary user-specified poses to drive the Blender model instantly, serving two purposes:

1. **Live end-to-end demonstration** — the user's physical hand is mirrored by the Blender model in real time, and the resulting mesh feeds directly into the Sionna RF simulation for real-time CSI computation.
2. **Rapid dataset expansion** — the live capture mechanism can record arbitrary poses that would be tedious to define by hand, expanding the gesture vocabulary without new animation scripting.

## MediaPipe Hand Landmark Model

The [MediaPipe HandLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) returns 21 normalised 3D landmark coordinates per frame:

- Landmark 0: Wrist
- Landmarks 1–4: Thumb (CMC → tip)
- Landmarks 5–8: Index finger (MCP → tip)
- Landmarks 9–12: Middle finger (MCP → tip)
- Landmarks 13–16: Ring finger (MCP → tip)
- Landmarks 17–20: Pinky (MCP → tip)

The model file (`mediaPipe/hand_landmarker.task`) runs locally on CPU in real time on commodity hardware.

## Architecture

```
                    ┌─────────────┐       TCP socket        ┌───────────────────┐
  Webcam ──► OpenCV │  sender.py  │ ──── landmarks (JSON) ──► blender_receiver/  │
                    │ (MediaPipe  │       localhost:65432    │ reciever_v4.py    │
                    │  env)       │                          │ (Blender Python)  │
                    └─────────────┘                          └────────┬──────────┘
                                                                      │
                                                              bone rotation update
                                                              + PLY export trigger
                                                                      │
                                                                      ▼
                                                            hand_frames_live/*.ply
```

## Sender (`sender.py`)

1. Opens the default webcam with OpenCV.
2. Passes each frame to the MediaPipe HandLandmarker model.
3. Extracts the 21 normalised 3D landmark coordinates for the first detected hand.
4. Serialises the landmark list as JSON and sends it over a TCP socket to the Blender receiver.

Runs in the **MediaPipe Python environment** (`mediaPipe/mediaPipeEnv/`), which has `mediapipe` and `opencv-python` installed separately from the Sionna environment.

## Blender Receiver Versions

Four successive receiver versions are retained in `blender_receivers/`, each improving on the previous:

### `reciever.py` (v1)
Basic TCP socket receiver. Reads landmark coordinates and directly sets bone rotations to fixed angles derived from crude finger-open / finger-closed logic. No palm coordinate frame; curl estimation is coarse.

### `reciever_v2.py` (v2)
Adds a palm coordinate frame. Constructs a local frame from the wrist landmark (index 0) and two palm anchor points (indices 5 and 17), then projects finger knuckle vectors onto this frame to compute per-finger curl angles. More anatomically consistent across different hand orientations.

### `reciever_v3.py` (v3)
Adds temporal smoothing using an exponential moving average (EMA) on the computed bone angles. Suppresses landmark jitter from the MediaPipe model, producing smoother armature animation. Thumb handling is improved with a dedicated projection axis.

### `reciver_v4.py` (v4 — production)
Full implementation used in the live demo:
- Palm coordinate frame construction from wrist + palm anchor landmarks.
- Per-finger curl angle computed by projecting the base-to-tip knuckle vector onto the palm plane.
- Thumb curl uses a separate local coordinate frame aligned with the thumb metacarpal axis.
- Spread angle (MCP lateral spread) estimated from the inter-finger base landmark distance.
- EMA temporal smoothing with configurable window size.
- `bpy.app.timers` callback at 10 ms intervals — non-blocking Blender UI integration.
- `.ply` export triggered after each armature update for downstream RF simulation.

## MediaPipe-to-Blender Bone Mapping

| MediaPipe Landmarks | Blender Bone | DOF Driven |
|---------------------|--------------|-----------|
| 0 (wrist) | `wrist` | Global position reference |
| 5, 9, 13, 17 (MCP bases) | `palm_*` | Spread (Z rotation) |
| 5→6→7→8 (index chain) | `index_01/02/03` | Curl (X rotation) |
| 9→10→11→12 (middle chain) | `middle_01/02/03` | Curl (X rotation) |
| 13→14→15→16 (ring chain) | `ring_01/02/03` | Curl (X rotation) |
| 17→18→19→20 (pinky chain) | `pinky_01/02/03` | Curl (X rotation) |
| 1→2→3→4 (thumb chain) | `thumb_01/02` | Curl + spread |

## Live Demo Results

- The live Blender hand accurately mirrors the user's physical hand across all 8 gesture classes.
- Landmark-to-bone update latency is approximately 10 ms (one timer tick).
- Open palm, fist, thumbs-up, V-sign, point, and pinch gestures were all demonstrated to replicate correctly.
- Rapid wrist rotation is tracked but introduces more pose ambiguity (as also seen in Case 3 of the ML experiments).

## Directory Structure

```
06_mediapipe_integration/
├── sender.py                        # MediaPipe landmark sender (production)
├── blender_receivers/
│   ├── reciever.py                  # v1: basic receiver
│   ├── reciever_v2.py               # v2: palm coordinate frame
│   ├── reciever_v3.py               # v3: + temporal smoothing
│   └── reciver_v4.py                # v4: full production receiver
└── mediaPipe/
    ├── hand_landmarker.task          # MediaPipe HandLandmarker model file
    ├── reloc.pdf                     # Reference document
    └── mediaPipeEnv/                 # MediaPipe Python environment (excluded from Git)
```

## Setup

### MediaPipe Environment

```bash
# Create a dedicated environment (Python 3.10–3.12 recommended for MediaPipe)
python3 -m venv mediaPipe/mediaPipeEnv
source mediaPipe/mediaPipeEnv/bin/activate
pip install mediapipe opencv-python
```

### Running the Sender

```bash
source mediaPipe/mediaPipeEnv/bin/activate
python sender.py
```

The sender listens on `localhost:65432` by default. Start the Blender receiver (inside Blender's Scripting workspace) before starting the sender.

## Dependencies

| Package | Used in |
|---------|---------|
| `mediapipe` | sender.py — HandLandmarker model |
| `opencv-python` | sender.py — webcam capture |
| `bpy` | blender_receivers/ — Blender embedded Python |
| `socket`, `json` | both sides — TCP communication |
