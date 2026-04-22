# 02 — Hand Parameterization

## What This Is

Hand parameterization is about choosing a compact, consistent representation of a hand pose that can be used for ML training, comparison between poses, and bridging between the Blender rig and MediaPipe's real-world hand detection.

---

## Two Representations Used in This Project

### Representation A — Bone Angles (Internal Blender Parameters)

The rig has ~27 degrees of freedom. A pose can be described as a vector of joint angles:

```
[mcp_curl_index, mcp_spread_index,
 pip_index, dip_index,
 mcp_curl_middle, mcp_spread_middle,
 pip_middle, dip_middle,
 ...
 thumb_x, thumb_y, thumb_z, thumb_tip,
 palm_index, palm_middle, palm_ring, palm_pinky, palm_thumb,
 wrist_x, wrist_y, wrist_z,
 global_rot_x, global_rot_y, global_rot_z]
```

**Advantage:** Compact (~27–35 values), directly controls the rig, good for procedural generation of new poses.  
**Disadvantage:** Not directly compatible with MediaPipe output. Requires a mapping step.

### Representation B — 21 Joint World Positions (MediaPipe-Compatible)

Each of the 20 bones contributes one joint position (the bone's head in world space). With wrist added, this gives 21 points matching MediaPipe's skeleton:

```
wrist
palm_thumb, palm_index, palm_middle, palm_ring, palm_pinky
thumb_01, thumb_02
index_01, index_02, index_03
middle_01, middle_02, middle_03
ring_01, ring_02, ring_03
pinky_01, pinky_02, pinky_03
```

Each joint = (x, y, z) → total = 21 × 3 = 63 values.

**Advantage:** Directly compatible with MediaPipe output. Easy to visualize. Velocity (Δposition/frame) is meaningful for gesture dynamics.  
**Disadvantage:** Larger feature vector. Position values change with global hand location, so wrist-relative coordinates are also stored.

---

## What Is Stored in the JSON

`hand_motion_full.json` and `hand_keypoints_6dof.json` store per-frame data with both representations:

```json
{
  "frame": 42,
  "global": {
    "position": [x, y, z],
    "rotation_quaternion": [w, x, y, z]
  },
  "joints": {
    "wrist": {
      "local_position": [...],
      "world_position": [...],
      "rotation_euler": [...],
      "rotation_quaternion": [...],
      "velocity": [dx, dy, dz]
    },
    "index_01": { ... },
    ...
  },
  "global_velocity": [dx, dy, dz]
}
```

---

## Dependency Rules for Realistic Poses

When generating random poses, the middle (PIP) and tip (DIP) joints are not sampled independently — they are computed from the base (MCP) curl and palm cupping to maintain anatomical plausibility:

```python
MID_FACTOR            = 0.68   # PIP ≈ 68% of MCP curl
TIP_FACTOR            = 0.6    # DIP ≈ 60% of PIP
PALM_INFLUENCE_ON_MID = 0.25   # palm cupping adds to PIP
PALM_INFLUENCE_ON_TIP = 0.12   # palm cupping adds to DIP
```

So: `pip = mcp_curl * 0.68 + palm_cup * 0.25`, `dip = pip * 0.6 + palm_cup * 0.12`

This prevents unrealistic poses like a finger where the base is straight but the tip is fully curled.

---

## Joint Angle Limits (Degrees)

| Joint | Min | Max |
|-------|-----|-----|
| MCP curl | -20 | 90 |
| MCP twist | -20 | 20 |
| MCP spread | -30 | 30 |
| PIP (mid) | 0 | 110 |
| DIP (tip) | 0 | 100 |
| Thumb base X | -30 | 60 |
| Thumb base Y | -50 | 50 |
| Thumb base Z | -50 | 50 |
| Thumb tip | 0 | 90 |
| Palm | -30 | 40 |

---

## MediaPipe Compatibility

MediaPipe outputs 21 hand landmark positions normalized to [0,1] in the image frame, with depth estimated. Our Blender representation uses 20 bone head positions in world space. The mapping is:

| MediaPipe | Blender |
|-----------|---------|
| Landmark 0 (wrist) | `wrist` bone head |
| Landmarks 1–4 (thumb) | `palm_thumb`, `thumb_01`, `thumb_02`, + tip |
| Landmarks 5–8 (index) | `palm_index`, `index_01`, `index_02`, `index_03` |
| Landmarks 9–12 (middle) | `palm_middle`, `middle_01`, `middle_02`, `middle_03` |
| Landmarks 13–16 (ring) | `palm_ring`, `ring_01`, `ring_02`, `ring_03` |
| Landmarks 17–20 (pinky) | `palm_pinky`, `pinky_01`, `pinky_02`, `pinky_03` |

---

## ML Bridge (In Progress)

The goal is to train a model:

```
MediaPipe 21-point coords  →  [model]  →  Blender bone angles
```

This would allow live MediaPipe data to drive the Blender rig without the fluctuations and mirroring issues of direct coordinate mapping. The training data would come from `random_pose_ML.py` which generates (pose parameters, rendered image) pairs, but the rendered image → MediaPipe detection step is currently blocked because MediaPipe doesn't detect hands in synthetic low-poly renders.
