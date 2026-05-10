# RF-Based Hand Gesture Recognition — Synthetic Pipeline

**IIT Madras · M.Tech Project (MTP-2) · May 2026**  
**Author:** Katravath Dinesh Naik (CS24M018)  
**Guide:** Prof. Ayon Chakraborty, Dept. of CSE, IIT Madras

---

## Overview

This repository contains a fully synthetic end-to-end pipeline for contactless hand gesture recognition using radio-frequency (RF) signals. The key idea is that different hand poses scatter a mmWave signal differently, producing measurably distinct CSI/CIR signatures at the receiver — and that a machine learning model can recover the hand pose from these signatures alone, with no camera at inference time.

Because collecting real-world RF measurements for many hand poses is slow and environment-specific, the pipeline generates all training data synthetically:

1. A physically accurate 3D hand model in Blender is animated into canonical gestures and exported as per-frame `.ply` polygon meshes.
2. Each mesh is placed in a 28 GHz mmWave indoor room scene and ray-traced by NVIDIA Sionna to produce labelled CSI/CIR vectors.
3. Five classical ML classifiers are trained on the synthetic dataset and evaluated across three difficulty levels.
4. A live demo closes the loop: MediaPipe extracts hand landmarks from a webcam, drives the Blender model in real time, and feeds the resulting mesh frames into a parallel Sionna worker pool for live CSI computation.

**Project report:** [docs/CS24M018_MTP2_V2.pdf](docs/CS24M018_MTP2_V2.pdf)

---

## Key Results

| Dataset | Condition | Best Accuracy | Best Model |
|---------|-----------|--------------|------------|
| Case 1 | Static poses, fixed hand position | **100%** | XGBoost (magnitude features) |
| Case 2 | Static poses, random hand position | **86%** | XGBoost (temporal window features) |
| Case 3 | Static poses, fast wrist rotation | **59%** | XGBoost (temporal window features) |

t-SNE projections confirm that distinct gesture classes form compact, non-overlapping clusters in the RF feature space under fixed-position conditions, validating the core hypothesis that RF signals carry sufficient information for pose discrimination in simulation.

---

## Gesture Classes

8 canonical gesture classes are used for ML experiments:

| Gesture | Description |
|---------|-------------|
| `FIST` | All fingers fully curled |
| `OPEN` | Fully extended open palm |
| `PINCH` | Thumb and index finger touching |
| `POINT` | Index finger extended, others curled |
| `ROCK` | Index and pinky extended (horns) |
| `THUMBS_UP` | Thumb extended upward |
| `THUMBS_DOWN` | Thumb extended downward |
| `V_SIGN` | Index and middle fingers extended |

---

## Repository Structure

```
mtp/
├── 01_blender_gesture_modeling/   # 3D hand model, armature, gesture animation, PLY export
│   ├── assets/source2/            # Blender project file and hand mesh
│   ├── scripts/
│   │   ├── animation/             # Gesture dataset generation, random pose animation
│   │   ├── core/                  # Shared rig config, gesture definitions, utilities
│   │   ├── export/                # PLY frame exporter
│   │   └── reconstruction/        # Pose replay and verification scripts
│   └── preprocess_frames.py       # Scale normalisation and Tx/Rx placement
├── 02_rf_simulation/              # Sionna 28 GHz mmWave RF simulation
│   ├── scripts/                   # Single-process and parallelised CSI generation
│   ├── exploration_scripts/       # Empty-space / floor-only / full-room scene studies
│   ├── scene_configs/             # Mitsuba XML scene template
│   └── data/                      # Generated RF outputs and sample data
├── 03_hand_mesh_pipeline/         # PLY cleaning, normalisation, RF-to-pose alignment
├── 04_ml_classification/          # ML training, evaluation, and saved results
│   ├── scripts/                   # KNN/SVM/RF/XGBoost/CNN classifiers
│   └── results/                   # Confusion matrices, classification reports, t-SNE plots
├── 05_live_demo/                  # Real-time end-to-end demo pipeline
│   └── live_rf_plot_hand/         # Sender, Sionna worker pool, live CSI plotter
├── 06_mediapipe_integration/      # MediaPipe landmark sender + Blender receiver versions
│   ├── blender_receivers/         # Receiver v1–v4 with progressive improvements
│   └── mediaPipe/                 # HandLandmarker model file
├── 07_visualization/              # Images and videos of results
│   ├── images/                    # Gesture renders, Sionna CSI/CIR plots, MediaPipe demo
│   └── videos/                    # CSI/CIR demo videos for each scene configuration
├── docs/
│   └── CS24M018_MTP2_V2.pdf       # Full project report
├── requirements.txt
└── CITATION.cff
```

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| 3D modelling and animation | Blender 4/5 (Python `bpy` API) |
| Hand landmark detection | MediaPipe HandLandmarker (21 landmarks) |
| RF simulation | NVIDIA Sionna v1.2.2 + Mitsuba 3 ray tracing |
| Classical ML | scikit-learn, XGBoost |
| Deep learning | TensorFlow / Keras |
| Data formats | `.ply`, `.npz`, `.json` |
| Language | Python 3.10+ |

---

## RF Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Carrier frequency | 28 GHz |
| Bandwidth | 400 MHz |
| Subcarriers | 256 |
| Ray count | 50,000 per frame |
| Reflection depth | 6 |
| Hand material (εr) | 17.3 (ITU-R P.2040 biological tissue) |
| Hand material (σ) | 25.6 S/m |
| Scene | Full room — floor + 4 walls + ceiling |

A full room enclosure is essential: hand-only and floor-only scenes produce featureless CSI. The surrounding walls create a rich multipath environment that amplifies the hand's scattering contribution relative to the line-of-sight path.

---

## Quick Start

### 1. Generate gesture animation and export PLY frames (inside Blender)

```python
# In Blender Scripting workspace:
# Run scripts/animation/gesture_dataset.py to generate 10,000-frame animation
# Run scripts/export/export_frames.py to export one .ply per frame
```

### 2. Clean and normalise mesh frames

```bash
python 03_hand_mesh_pipeline/clean_frames.py
python 01_blender_gesture_modeling/preprocess_frames.py
```

### 3. Run RF simulation

```bash
source 02_rf_simulation/sionna_env/bin/activate
python 02_rf_simulation/scripts/save_rf_parallel_v2.py
```

### 4. Align RF output with pose labels

```bash
python 03_hand_mesh_pipeline/link_rf_to_blender.py
```

### 5. Train and evaluate classifiers

```bash
python 04_ml_classification/scripts/ml_model.py
```

### 6. Run the live demo

```bash
# Start Blender, load hand_Lowpoly.blend, run blender_receivers/reciver_v4.py in Scripting workspace
# Then in a terminal:
source 02_rf_simulation/sionna_env/bin/activate
python 05_live_demo/live_rf_plot_hand/run_pipeline.py
```

---

## Notes on Generated Data

Large generated files are excluded from Git and must be regenerated locally:

- `.ply` frame sequences — export from Blender, then run `03_hand_mesh_pipeline/clean_frames.py`
- Pose `.json` logs — produced alongside PLY export by the animation scripts
- `.npz` RF archives — produced by `02_rf_simulation/scripts/save_rf_parallel_v2.py`

---

## Citation

If you use this work, please cite the project report or use the metadata in [CITATION.cff](CITATION.cff).
