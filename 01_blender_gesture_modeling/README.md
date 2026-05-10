# 01 Blender Gesture Modeling

This module contains Blender-side assets and scripts for hand-model preparation, gesture animation, frame export, and motion reconstruction.

The underlying rig follows a low-poly hand model with a dedicated armature that supports finger curl, spread, thumb articulation, palm motion, and wrist rotation for simulation-ready animation.

## Structure

- `assets/`: archived hand models, Blender source material, and linked asset bundles.
- `frame_preview.py`, `frame_preview_v2.py`: mesh-frame inspection helpers.
- `hand_check.py`: validation helper for hand mesh integrity.
- `preprocess_frames.py`: preprocessing utility for exported frame sequences.
- `scripts/animation/`: gesture-dataset generation, random-pose synthesis, and animation scripts.
- `scripts/core/`: shared rig configuration, gesture definitions, and utility modules.
- `scripts/export/`: Blender export helpers for mesh frames and script extraction.
- `scripts/reconstruction/`: reconstruction scripts driven by stored pose or keypoint JSON.

## Modeling Notes

- Finger motion primarily uses curl on the local X axis, with spread handled on Z.
- Shared configuration in `scripts/core/` defines finger groups, curl weights, and bias terms used to keep poses anatomically natural.
- Reconstruction scripts provide a validation path from saved JSON back into Blender animation.

## Generating Outputs in Blender

Typical local workflow:

1. Open the hand rig `.blend` asset from `assets/source2/`.
2. Run one of the scripts in `scripts/animation/` from Blender's scripting workspace to create gesture motion.
3. Export mesh frames with `scripts/export/export_frames.py`.
4. Generate motion JSON from the animation or reconstruction scripts when downstream linking or validation requires it.

Large generated `.json` motion files and exported `.ply` frame datasets are intentionally kept out of Git and should be regenerated locally.
