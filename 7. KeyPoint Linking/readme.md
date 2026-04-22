# 07 — Keypoint Linking

## What This Is

Scripts that pair the RF data (CSI/CIR from Sionna) with the ground-truth hand keypoints (joint positions from Blender's JSON) into a single combined dataset ready for ML training.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `link_rf_to_blender.py` | Main linking script — builds and saves the combined NPZ |
| `load_npz.py` | Inspection script — loads and prints the linked dataset |

---

## The Index Offset Problem

PLY files are 0-indexed: `hand_0000.ply`, `hand_0001.ply`, ...  
JSON keypoint frames are 1-indexed: `frame: 1`, `frame: 2`, ...

So: `json_frame = ply_frame_idx + 1`

`link_rf_to_blender.py` handles this automatically.

---

## What Gets Linked Per Frame

For each `rf_frame_NNN.npz`:

| Field | Source | Description |
|-------|--------|-------------|
| `ply_frame_idx` | NPZ filename | 0-based PLY index |
| `json_frame` | ply_idx + 1 | 1-based JSON frame number |
| `H` | rf_frame_NNN.npz | CSI vector (256,) complex |
| `CIR` | rf_frame_NNN.npz | CIR vector (256,) complex |
| `keypoints` | hand_motion_full.json | All 20 joints: world_pos, local_pos, rotation |
| `global` | hand_motion_full.json | Arm position + rotation quaternion |
| `global_velocity` | hand_motion_full.json | Velocity of arm root |

---

## Output Dataset Structure

`linked_all_frames.npz` contains:

```
H_mat              : (N, 256)     complex64   — CSI per frame
CIR_mat            : (N, 256)     complex64   — CIR per frame
world_positions    : (N, 20, 3)   float32     — joint positions per frame
local_positions    : (N, 20, 3)   float32     — local joint positions
rotations_quat     : (N, 20, 4)   float32     — joint rotations
joint_names        : (20,)        str         — name of each joint
ply_frame_indices  : (N,)         int32       — PLY file indices
json_frame_numbers : (N,)         int32       — JSON frame numbers
rf_files           : (N,)         str         — source NPZ filenames
ply_files          : (N,)         str         — source PLY filenames
freq_axis          : (256,)       float32     — subcarrier frequencies
```

---

## How to Load

```python
import numpy as np

data = np.load("linked/linked_all_frames.npz", allow_pickle=True)

H_mat           = data["H_mat"]             # (N, 256) complex64
CIR_mat         = data["CIR_mat"]           # (N, 256) complex64
world_positions = data["world_positions"]   # (N, 20, 3) float32
joint_names     = data["joint_names"]       # (20,) array of strings

# Get wrist position for frame 5
wrist_col = list(joint_names).index("wrist")
wrist_pos = world_positions[5, wrist_col]   # (3,)

# Get CSI for frame 5
H_frame5 = H_mat[5]                         # (256,) complex
```

---

## Running the Linker

```bash
python link_rf_to_blender.py
```

The script prompts:
```
Link ALL frames or a range? [all / range] :
```

- Type `all` to link everything in `rf_output/`
- Type `range` to specify start and end PLY frame indices

Output is saved to `linked/linked_all_frames.npz` or `linked/linked_frames{start}-{end}.npz`.

---

## Checking for Missing Data

The script prints warnings if any frames are missing keypoints or PLY files:
```
WARNING: 3 frames missing keypoints : [5, 6, 7]
WARNING: 1 frames missing PLY file  : [0]
```

These gaps can occur if the animation length (JSON) and number of exported PLY frames don't match. The linked dataset fills missing entries with zeros.

---

## Directory Layout Expected

```
hand_models/
├── hand_frames_normalized/    ← PLY files (input)
├── rf_output/                 ← rf_frame_NNN.npz files (input)
├── source 2/
│   └── hand_motion_full.json  ← keypoints JSON (input)
└── linked/
    └── linked_all_frames.npz  ← output
```
