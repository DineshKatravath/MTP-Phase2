# 03 Hand Mesh Pipeline

This module stores the exported hand-mesh datasets and the utilities used to clean, normalize, and align them with RF samples.

## Structure

- `clean_frames.py`: mesh-cleaning utility.
- `link_rf_to_blender.py`: helper for aligning RF samples with Blender frame indices.
- `raw_frames/`: raw mesh exports from Blender.
- `cleaned_frames/`: quality-filtered frames.
- `normalized_frames/`: normalized mesh frames used by downstream scripts.
- `wrist_frames/`: wrist-motion subset used in separate experiments.

## Pipeline Notes

- Mesh export indices and keypoint JSON indices are not always identical, so `link_rf_to_blender.py` serves as the alignment step between geometry and RF data.
- The normalized frame sets are the intended bridge between Blender export and Sionna simulation.

## Regeneration

- Export the raw `.ply` frames from Blender first.
- Place them into the expected raw-frame directory structure.
- Run the cleaning and normalization utilities locally before RF simulation.

The `.ply` datasets in this stage are treated as generated data and are not intended to be versioned in GitHub.
