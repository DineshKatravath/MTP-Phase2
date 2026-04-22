# 04 — Dataset Generation

## What This Is

Scripts for generating large numbers of diverse, anatomically plausible hand poses programmatically, along with their keypoint data. This is the source of training data diversity — without random poses, the dataset would only contain 5–6 gesture classes.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `random_pose.py` | Apply a single random pose to the rig (no animation, no export) |
| `random_pose_anim.py` | Generate a continuous random-pose animation over 240 frames |
| `gestures_anim_global_random.py` | Full-DOF random animation over 10,000 frames + keypoint JSON export |
| `random_pose_ML.py` | Random poses + camera-fitted renders + bone rotation export for ML |

---

## How Random Poses Are Generated

### Step 1 — Sample Base Parameters

For each finger, sample the MCP (base) joint angles independently:

```python
base_curl   = random.uniform(0, 95)      # main curl
base_twist  = random.uniform(-8, 8)      # slight twist
base_spread = random.uniform(-12, 12)    # fan spread
```

For thumb, wrist, palm, and global rotation, sample from their respective ranges.

### Step 2 — Enforce Dependencies

PIP (mid) and DIP (tip) are NOT sampled independently. They are derived from MCP:

```python
pip = clamp(mcp_curl * 0.68 + palm_cup * 0.25,  0, 110)
dip = clamp(pip      * 0.60 + palm_cup * 0.12,  0, 100)
```

This prevents anatomically impossible configurations (e.g. base straight, tip fully curled).

### Step 3 — Apply to Rig

Call `apply_gesture_full(g)` which sets bone rotations via `rotation_euler` or `rotation_quaternion`.

### Step 4 — Keyframe or Export

For animation: insert keyframes every `STEP = 40` frames.  
For ML dataset: render the current frame and save bone rotation data.

---

## Dataset Scale

The number of poses is configurable — set `NUM_POSES` in `random_pose_ML.py` or `FRAME_END` / `STEP` in `gestures_anim_global_random.py`. No fixed limit — generate as many as needed.

For Sionna processing, each unique pose = one frame → one PLY file → one RF snapshot. A 360-frame animation produces 360 data points.

---

## Angle Limits Reference

All random sampling stays within anatomically realistic bounds:

| Parameter | Range (degrees) |
|-----------|----------------|
| MCP curl | 0 to 95 |
| MCP twist | -8 to 8 |
| MCP spread | -12 to 12 (+ finger bias) |
| PIP (computed) | 0 to 110 |
| DIP (computed) | 0 to 100 |
| Palm cup | -6 to 20 |
| Wrist X | -80 to 80 |
| Wrist Y | -60 to 60 |
| Wrist Z | -180 to 180 |
| Global rotation | -180 to 180 (all axes) |
| Thumb base X | -15 to 35 |
| Thumb base Y | -35 to 35 |
| Thumb base Z | -30 to 30 |
| Thumb tip | 0 to 60 |

---

## `random_pose_ML.py` — Special Features

This script adds a camera that auto-fits to the hand bounding box each frame so the hand fills the rendered image regardless of orientation. It renders to PNG and saves bone rotations to `rotations.json`.

The intended use: pair each rendered image with its bone rotations to train a model `image → bone angles`. This is not yet complete because MediaPipe doesn't detect the synthetic hand in renders, but the bone rotation data is still useful for direct coordinate-to-angle mapping.

---

## Output Files

| File | Contents |
|------|---------|
| `hand_motion_full.json` | Full bone data (position, rotation, velocity) per frame |
| `hand_keypoints_6dof.json` | 21-point keypoints (global + wrist-relative) per frame |
| `rotations.json` | Bone rotation quaternions for ML poses |
| `hand_dataset/frames/frame_NNNNN.png` | Rendered images of each pose |
