# 09 — Reconstruction

## What This Is

Scripts that rebuild a Blender animation from the saved JSON keypoint data. This serves as a validation step — if the reconstruction matches the original animation, it confirms the keypoint data is complete and accurate enough to fully describe each hand pose.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scene_reconstruction.py` | Rebuild animation from `hand_motion_full.json` (full bone rotation data) |
| `21rep_reconstruction.py` | Rebuild animation from `hand_keypoints_6dof.json` (21-point positions only) |

---

## `scene_reconstruction.py` — Full Reconstruction

Uses the complete bone rotation quaternions saved in `hand_motion_full.json`. For each frame:

1. Sets global arm location and rotation quaternion
2. For each bone: applies saved rotation quaternion (or Euler if quaternion unavailable)
3. Inserts keyframes for all bones and the armature object

This is a near-perfect reconstruction because the full rotation data is saved — no approximation needed.

**Input:** `hand_motion_full.json`  
**Output:** Blender animation on the rig

---

## `21rep_reconstruction.py` — Position-Based Reconstruction

Uses only the 21 world-space joint positions from `hand_keypoints_6dof.json`. Since only positions are saved (not rotations), the bone angles must be computed from the geometry:

For each finger:
```python
v1 = p1 - p0   # palm → MCP vector
v2 = p2 - p1   # MCP → PIP vector
v3 = p3 - p2   # PIP → DIP vector

a1 = angle(v1, v2)   # MCP angle
a2 = angle(v2, v3)   # PIP angle

bone_MCP.rotation_euler[0] = a1 * 1.5
bone_PIP.rotation_euler[0] = a1
bone_DIP.rotation_euler[0] = a2
```

This is an approximation — the angle between consecutive bone vectors gives a good estimate of the joint rotation but is not exact because it doesn't account for the rig's rest pose orientation or bone constraints.

**Input:** `hand_keypoints_6dof.json`  
**Output:** Approximate Blender animation

---

## How to Run

Both scripts run inside Blender:

1. Open `.blend` file with the hand rig
2. Place the JSON file in the same directory as the `.blend` file (or update the path)
3. Go to Scripting workspace
4. Open and run the reconstruction script

The script sets `scene.frame_start` and `scene.frame_end` automatically based on the JSON data length.

---

## Why This Matters

The reconstruction validates the data pipeline in two ways:

1. **`scene_reconstruction.py`** confirms that the JSON faithfully captures all bone rotations. If the reconstructed animation looks identical to the original, the data is good.

2. **`21rep_reconstruction.py`** validates the 21-point representation. If position-only data can approximately reconstruct the motion, it confirms that the coordinate data is informative enough for ML models that work with joint positions rather than angles.

The reconstruction also provides a way to visually inspect the dataset — by scrubbing through the reconstructed animation in Blender, you can check that all frames look physically plausible before running the expensive Sionna simulation.

---

## Limitations of Position-Based Reconstruction

- Does not capture global hand rotation (no wrist quaternion in position-only format)
- Finger angles are approximate (computed from direction vectors, not from actual bone constraints)
- Palm cupping and spread are not well-recovered from positions alone
- Thumb reconstruction is especially approximate due to its complex 3-DOF base joint

For ground-truth quality reconstruction, use `scene_reconstruction.py` with the full JSON data.
