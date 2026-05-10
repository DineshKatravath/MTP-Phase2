# 01 — Blender Gesture Modeling

This module is the geometry front-end of the synthetic RF pipeline. It produces animatable 3D hand poses, exports them as per-frame `.ply` meshes for RF simulation, and records every pose as a 32-DOF JSON parameter vector for downstream label alignment.

## Background

RF-based hand gesture recognition requires large quantities of labelled (hand-pose, CSI) pairs. Collecting these from real radio hardware is slow and environment-specific. This module replaces real-world data collection with a physically accurate 3D hand model that can be scripted into thousands of distinct poses and exported frame-by-frame into the Sionna ray-tracing simulator.

## Hand Model

The base mesh is a low-polygon hand model deliberately chosen over a high-resolution scan for three reasons:
- fewer polygons means less rigging complexity and fewer deformation artefacts
- each frame must be exported as a `.ply` file, so a lightweight mesh reduces per-frame I/O overhead across thousands of exports
- Sionna's ray-tracer intersects every triangle per ray, so a denser mesh directly increases simulation cost

The mesh was imported into Blender, duplicate objects removed, and all transforms applied (`Ctrl+A → Rotation & Scale`) to set rotation to `(0,0,0)` and scale to `(1,1,1)` — a prerequisite for predictable bone orientation.

## Armature and Bone Hierarchy

A 20-bone armature was built in Blender's Edit Mode by extruding bones sequentially following the anatomical structure of the human hand. The hierarchy is:

```
wrist
└── palm_index / palm_middle / palm_ring / palm_pinky / palm_thumb   (metacarpals)
    └── finger_01  (MCP base — curl, twist, spread)
        └── finger_02  (PIP — curl only)
            └── finger_03  (DIP — curl only)
thumb_01  (3-axis base)
└── thumb_02  (curl only)
```

| Bone | Anatomical Role | Count | DOF |
|------|----------------|-------|-----|
| wrist | Wrist root | 1 | 3 |
| palm_* | Metacarpal (palm cup axis) | 5 | 1 each |
| *_01 (fingers) | MCP base (curl, twist, spread) | 4 | 3 each |
| *_02 (fingers) | PIP (curl only) | 4 | 1 each |
| *_03 (fingers) | DIP (curl only) | 4 | 1 each |
| thumb_01 | Thumb base (3-axis) | 1 | 3 |
| thumb_02 | Thumb tip (curl only) | 1 | 1 |

**Total: 20 bones, 32 degrees of freedom.**

Biomechanical joint constraints (Blender's per-bone Limit Rotation) enforce anatomically plausible ranges of motion so that scripted and random poses cannot produce configurations a real hand cannot achieve.

## 32-DOF Pose Parameterisation

Every bone's active rotation axes are catalogued into a 32-dimensional parameter vector of local Euler angles. This vector uniquely describes any hand pose and is logged per-frame to a JSON file alongside the global wrist position. A reconstruction script verifies that replaying the JSON record exactly recovers the original animation, confirming the parameterisation is lossless.

JSON log format (per frame):
```json
{
  "frame": 42,
  "gesture": "FIST",
  "wrist_pos": [x, y, z],
  "bones": {
    "wrist": [rx, ry, rz],
    "palm_index": [rx],
    ...
  }
}
```

## Gesture Library

Ten canonical gestures are implemented using a modular curl-and-spread abstraction:

| Gesture | Description |
|---------|-------------|
| `OPEN` | Fully extended fingers, open palm |
| `FIST` | All fingers fully curled |
| `PINCH` | Thumb and index finger touching |
| `POINT` | Index finger extended, others curled |
| `ROCK` | Index and pinky extended (horns) |
| `THUMBS_UP` | Thumb extended upward |
| `THUMBS_DOWN` | Thumb extended downward |
| `V_SIGN` | Index and middle fingers extended |
| `WAVE_LEFT` | Open palm, wrist rotated left |
| `WAVE_RIGHT` | Open palm, wrist rotated right |

For ML experiments, `WAVE_LEFT` and `WAVE_RIGHT` are merged into `OPEN` (identical finger configuration, different wrist orientation only), giving **8 effective gesture classes**.

## Dataset Animation

`scripts/animation/gesture_dataset.py` generates a continuous 10,000-frame animation by cycling through all gesture classes in round-robin order:

- **Hold segment**: `HOLD_STEP = 70` frames — armature held at target pose, labelled with gesture name
- **Transition segment**: `TRANSITION_STEP = 30` frames — Blender interpolates between two consecutive poses, labelled `transition` and excluded from classifier training

This yields approximately **700 labelled (non-transition) frames per class** across 8 classes.

## Directory Structure

```
01_blender_gesture_modeling/
├── assets/
│   └── source2/
│       ├── hand_Lowpoly.blend       # Blender project file with armature
│       ├── hand_Lowpoly.obj / .mtl  # OBJ export
│       ├── hand_Lowpoly.glb         # GLB export
│       └── hand_dataset/
│           └── rotations.json       # Recorded pose parameters
├── scripts/
│   ├── core/
│   │   ├── config.py        # Finger groups, curl weights, bias terms
│   │   ├── gestures.py      # Canonical gesture definitions
│   │   ├── modules.py       # Rig control helpers
│   │   └── utils.py         # Shared utilities
│   ├── animation/
│   │   ├── gesture_dataset.py           # Main 10,000-frame dataset generator
│   │   ├── gestures_anim_21rep.py       # 21-keypoint-driven animation
│   │   ├── gestures_anim_global.py      # Global-position animation
│   │   ├── gestures_anim_global_random.py  # Random global-position variant
│   │   ├── palm_fist_palm.py            # Palm↔fist transition script
│   │   ├── random_pose.py               # Single random pose generator
│   │   └── random_pose_anim.py          # Random pose animation loop
│   ├── export/
│   │   ├── export_frames.py  # PLY frame exporter (one .ply per animation frame)
│   │   └── export_files.py   # Miscellaneous export helpers
│   └── reconstruction/
│       ├── 21rep_reconstruction.py  # Reconstruct pose from 21 world-space joints
│       └── scene_reconstruction.py  # Replay full animation from JSON angles
├── frame_preview.py      # Inspect a single exported frame mesh
├── frame_preview_v2.py   # Updated frame inspection helper
├── hand_check.py         # Validate hand mesh integrity
└── preprocess_frames.py  # Preprocessing utility for exported frame sequences
```

## Typical Workflow

1. Open `assets/source2/hand_Lowpoly.blend` in Blender.
2. Open the Scripting workspace.
3. Run a script from `scripts/animation/` to generate gesture motion (e.g. `gesture_dataset.py` for the full 10,000-frame dataset).
4. Run `scripts/export/export_frames.py` to export one `.ply` file per frame into an output directory.
5. Run `scripts/reconstruction/scene_reconstruction.py` to verify that the JSON record exactly recovers the animation.
6. Pass the exported `.ply` directory to `03_hand_mesh_pipeline/` for cleaning and normalization before RF simulation.

> Large generated `.json` motion files and exported `.ply` frame datasets are intentionally excluded from Git and must be regenerated locally.

## Dependencies

- Blender 4.x or 5.x (scripts use `bpy` — Blender's embedded Python API)
- No external Python packages required inside Blender

## Key Results

- The 32-DOF parameterisation is verified lossless: JSON replay exactly recovers every recorded pose.
- World-space joint positions alone are sufficient to approximately recover any pose (demonstrated by `21rep_reconstruction.py`).
- 10,000-frame animations produce ~700 clean labelled frames per gesture class after transition filtering.
