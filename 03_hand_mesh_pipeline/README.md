# 03 — Hand Mesh Pipeline

This module bridges the Blender geometry export and the Sionna RF simulation stage. It cleans and normalises raw `.ply` mesh frames exported from Blender, and aligns each processed frame with its corresponding RF output using a frame-index linker.

## Background

Blender exports one `.ply` polygon mesh file per animation frame. These raw exports contain duplicate vertices, inconsistent scales, and coordinate systems that are incompatible with Sionna's scene geometry expectations. This module standardises the mesh files so that:

1. Sionna can reliably load and ray-trace each frame.
2. The frame index in the `.ply` filename exactly matches the frame index in the RF output and the pose JSON log, enabling accurate (pose, CSI) label alignment downstream.

## Pipeline Stages

```
Blender export (.ply per frame)
         │
         ▼
  raw_frames/              ← raw, uncleaned exports
         │
  clean_frames.py
         │
         ▼
  cleaned_frames/          ← duplicate-free, watertight meshes
         │
  (scale normalisation,
   Tx/Rx placement by
   preprocess_frames.py
   in module 01)
         │
         ▼
  normalized_frames/       ← scale-normalised, Sionna-ready meshes
         │
  link_rf_to_blender.py
         │
         ▼
  linked .npz archive      ← (CSI, CIR, pose_label) aligned by frame index
```

## Scripts

### `clean_frames.py`

Reads each raw `.ply` file from `raw_frames/` and applies:

- **Duplicate vertex removal** — merges vertices within a tolerance threshold to produce watertight meshes that Sionna's ray-tracer can intersect correctly.
- **Degenerate face removal** — discards triangles with zero area that would cause numerical issues in the ray-tracer.
- **Normal recomputation** — recalculates surface normals after cleaning so that material reflections are physically accurate.

Cleaned meshes are written to `cleaned_frames/hand_frames_clean/`.

### `link_rf_to_blender.py`

The frame index embedded in the `.ply` filename (e.g. `hand_0042.ply`) and the frame index in the Sionna output are not always identical because:
- Blender's frame counter may not start at zero.
- Parallel Sionna workers may complete frames out of order.
- Transition frames may be dropped from the RF output but retained in the pose JSON.

This script constructs a frame-index mapping between the RF output arrays and the Blender pose JSON, producing an aligned `.npz` archive where each row is a matched `(CSI, CIR, gesture_label)` triple. This archive is the direct input to the ML classification stage.

## Directory Structure

```
03_hand_mesh_pipeline/
├── clean_frames.py         # Mesh cleaning utility
├── link_rf_to_blender.py   # RF-to-pose frame alignment
├── raw_frames/             # Raw .ply exports from Blender (not versioned)
├── cleaned_frames/
│   └── hand_frames_clean/  # Cleaned, watertight .ply meshes
├── normalized_frames/      # Scale-normalised Sionna-ready meshes (not versioned)
└── wrist_frames/           # Wrist-rotation subset used in Case 3 experiments
```

> `.ply` datasets are treated as generated data and are not versioned in Git. Regenerate locally by running the Blender export followed by `clean_frames.py`.

## Regeneration Steps

1. Export raw `.ply` frames from Blender using `01_blender_gesture_modeling/scripts/export/export_frames.py`.
2. Place the exported directory as `raw_frames/` (or update the input path in `clean_frames.py`).
3. Run `clean_frames.py` to produce `cleaned_frames/hand_frames_clean/`.
4. Run `preprocess_frames.py` (from `01_blender_gesture_modeling/`) on the cleaned frames to normalise scale and compute Tx/Rx positions, producing `normalized_frames/`.
5. Pass `normalized_frames/` to the Sionna scripts in `02_rf_simulation/` for CSI generation.
6. After CSI generation, run `link_rf_to_blender.py` with the RF output `.npz` and the pose JSON to produce the linked archive for ML training.

## Linked Archive Format

The output of `link_rf_to_blender.py` is a `.npz` file containing:

| Array | Description |
|-------|-------------|
| `H_mat` | Complex CSI matrix, shape `(N, 256)` |
| `CIR_mat` | Complex CIR matrix, shape `(N, D)` |
| `gestures` | String gesture labels, shape `(N,)` |
| `frame_ids` | Original Blender frame indices, shape `(N,)` |
| `wrist_pos` | Global wrist positions from JSON log, shape `(N, 3)` |

Transition frames (label contains `"transition"`) are included in the archive but filtered out before classifier training in `04_ml_classification/`.

## Dependencies

- Python 3.x
- NumPy
- Open3D or trimesh (for PLY cleaning)

```bash
pip install open3d
# or
pip install trimesh
```
