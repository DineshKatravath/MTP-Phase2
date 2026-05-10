# 02 RF Simulation

This module contains the NVIDIA Sionna stage of the project, including scene descriptions, RF generation scripts, and experiment outputs.

## Structure

- `scripts/`: main RF simulation runners and support utilities.
- `exploration_scripts/`: scene-variation experiments such as empty-room, floor-only, and walls-plus-floor studies.
- `scene_configs/`: XML and scene assets used by Sionna.
- `data/`: generated RF outputs and reference datasets.
- `sionna_env/`: local experiment environment retained for reproducibility but excluded from Git tracking.

## Simulation Notes

- The main experiments use 28 GHz carrier frequency, 400 MHz bandwidth, and 256 subcarriers.
- RF scripts compute CSI first and derive CIR from the complex frequency response.
- Exploration scripts compare environmental configurations such as empty space, floor-only, and walls-plus-floor scenes.
