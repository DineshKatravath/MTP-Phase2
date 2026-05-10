# RF-Based Hand Gesture Recognition using Blender, MediaPipe, and Sionna

**IIT Madras · M.Tech Project (MTP-2) · 2025-26**  
**Roll No:** CS24M018

## Overview

This repository contains a simulation-driven pipeline for contactless hand-gesture recognition using RF sensing. The workflow combines Blender-based hand modeling, MediaPipe landmark capture, NVIDIA Sionna ray-tracing, and downstream machine-learning models for six gesture classes.

The project flow is:

1. Hand gestures are modeled and animated in Blender.
2. Per-frame hand meshes are exported as `.ply` sequences.
3. Sionna simulates RF propagation and produces CSI/CIR outputs.
4. Classical ML and deep-learning models classify gestures from simulated RF data.
5. A live demo uses MediaPipe landmarks to drive the Blender hand and trigger the RF pipeline in real time.

The broader research goal is to study how well simulated mmWave RF signals can differentiate hand gestures and support downstream gesture-recognition models.

## Repository Structure

```text
mtp/
├── 01_blender_gesture_modeling/
│   ├── assets/
│   ├── scripts/
│   │   ├── animation/
│   │   ├── core/
│   │   ├── export/
│   │   └── reconstruction/
│   ├── frame_preview.py
│   ├── frame_preview_v2.py
│   ├── hand_check.py
│   └── preprocess_frames.py
├── 02_rf_simulation/
│   ├── scripts/
│   ├── exploration_scripts/
│   ├── scene_configs/
│   └── data/
├── 03_hand_mesh_pipeline/
├── 04_ml_classification/
├── 05_live_demo/
│   └── live_rf_plot_hand/
├── 06_mediapipe_integration/
│   ├── blender_receivers/
│   └── mediaPipe/
├── 07_visualization/
│   ├── images/
│   └── videos/
└── docs/
```

## Gesture Labels

| Gesture | Label |
| --- | --- |
| Palm | 0 |
| Fist | 1 |
| Thumbs Up | 2 |
| Thumbs Down | 3 |
| OK | 4 |
| Point | 5 |

## Main Modules

- `01_blender_gesture_modeling/`: Blender assets, animation scripts, export helpers, and reconstruction utilities.
- `02_rf_simulation/`: Sionna scene setup, RF simulation scripts, experiment variants, and generated RF outputs.
- `03_hand_mesh_pipeline/`: raw, cleaned, normalized, and wrist-motion mesh-frame datasets.
- `04_ml_classification/`: feature extraction, baseline ML models, CNN variants, and saved evaluation outputs.
- `05_live_demo/`: real-time runtime pipeline, RF plotting, and live Sionna workers.
- `06_mediapipe_integration/`: MediaPipe sender and Blender receiver variants for landmark-driven control.
- `07_visualization/`: figures and videos used for documentation and presentation.
- `docs/`: formal project report.

## Technology Stack

| Component | Technology |
| --- | --- |
| 3D Modelling and Animation | Blender |
| Hand Landmark Detection | MediaPipe Hands |
| RF Simulation | NVIDIA Sionna |
| Classical ML | scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Data Formats | `.ply`, `.npy`, `.npz` |
| Language | Python 3.10+ |

## RF Configuration

The main RF experiments use a 28 GHz mmWave setup with 400 MHz bandwidth and 256 subcarriers. CSI is generated in Sionna and CIR is obtained through IFFT-based post-processing. The working setup places a single TX and RX around the hand mesh inside controlled room-like scenes with floor and wall reflectors.

## Quick Start

### RF Simulation

```bash
python 02_rf_simulation/scripts/save_rf_data.py
```

### Classification

```bash
python 04_ml_classification/scripts/ml_model.py
```

### Live Demo

```bash
python 06_mediapipe_integration/sender.py
python 05_live_demo/live_rf_plot_hand/run_pipeline.py
```

## Regenerating Blender Outputs

The repository does not need committed raw `.ply` frame sequences or large Blender-exported motion `.json` files. These are generated locally from the Blender stage when needed.

- Use the animation scripts under `01_blender_gesture_modeling/scripts/animation/` to create or keyframe gesture motion.
- Use `01_blender_gesture_modeling/scripts/export/export_frames.py` inside Blender to export per-frame hand meshes.
- Use the reconstruction and animation scripts to regenerate motion JSON outputs from Blender when required by downstream steps.
- Use the mesh-processing utilities in `03_hand_mesh_pipeline/` before running RF simulation.

## Report

The full project report is available at [docs/CS24M018_MTP2_V2.pdf](/Users/dinesh/Documents/mtp/docs/CS24M018_MTP2_V2.pdf).

## Citation

If you use this work, please cite the M.Tech project report or the metadata in [CITATION.cff](/Users/dinesh/Documents/mtp/CITATION.cff).
