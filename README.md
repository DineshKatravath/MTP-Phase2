# MTP — Hand Gesture Recognition via RF Sensing

## Project Overview

This project builds a simulation pipeline that generates mmWave RF signals (CSI/CIR) from animated 3D hand gestures, with the goal of training a model that can **optimally differentiate between gestures** a person performs and **suggest the best personalized gesture vocabulary** for that individual based on RF distinguishability.

The core idea: use Blender to generate realistic hand gesture animations → export per-frame meshes → run mmWave ray-tracing in Sionna → collect CSI/CIR data → pair with ground-truth joint keypoints → train a gesture recognition/optimization model.

---

## Final Pipeline (End-to-End)

```
Blender Hand Rig
      │
      ▼
Gesture Animation (Python scripts in Blender)
      │
      ▼
Per-frame PLY mesh export
      │
      ▼
PLY cleaning + normalization to meters (trimesh)
      │
      ▼
Sionna Ray-Tracing Scene (28 GHz mmWave, floor + walls)
  TX ──── hand mesh ──── RX
      │
      ▼
CSI (Channel Frequency Response) + CIR (Channel Impulse Response)
      │
      ▼
RF data saved as .npz per frame
      │
      ▼
Ground-truth keypoints saved as JSON from Blender
      │
      ▼
Linked dataset: RF + keypoints → linked_all_frames.npz
      │
      ▼
Spectrograms (slow-time × subcarrier/delay-tap matrix)
      │
      ▼
[Future] Model training for gesture optimization
```

---

## RF System Parameters

| Parameter | Value |
|-----------|-------|
| Carrier Frequency | 28 GHz (mmWave) |
| Bandwidth | 400 MHz |
| Subcarriers | 256 |
| CIR Taps | 256 (= IFFT of CSI) |
| Delay Resolution (Δτ) | 2.5 ns (= 1/BW) |
| TX Antennas | 1 (isotropic, vertical polarization) |
| RX Antennas | 1 (isotropic, vertical polarization) |
| TX Position | [-0.143, 0.020, -0.020] m |
| RX Position | [0.167, 0.020, -0.020] m |
| Propagation Modes | LOS + specular reflection + diffuse reflection + refraction + diffraction |
| Max Bounce Depth | 6 |
| Mitsuba Variant | `llvm_ad_mono_polarized` |
| Hand EM Properties | εᵣ = 17.3, σ = 25.6 S/m (skin at 28 GHz) |

---

## Blender Rig Parameters

| Parameter | Value |
|-----------|-------|
| Mesh | Low-poly hand (downloaded from Sketchfab) |
| Total Bones | 20 |
| Finger Bones | 3 per finger (index, middle, ring, pinky) = 12 |
| Thumb Bones | 2 |
| Palm Bones | 5 (one per finger ray) |
| Wrist Bone | 1 |
| Finger DOF | Base bone: 2 DOF (curl + spread), Mid/Tip: 1 DOF each |
| Palm DOF | 1 DOF each |
| Wrist DOF | 3 DOF |

---

## Folder Structure

```
MTP_Hand_RF_Gesture/
├── README.md                        ← This file
├── 00_project_evolution/            ← How the project evolved, decisions, lessons
├── 01_hand_modeling/                ← Blender rig setup and bone structure
├── 02_hand_parameterization/        ← Representing hand as parameter vectors
├── 03_gesture_animation/            ← Python scripts for animating gestures
├── 04_dataset_generation/           ← Random pose generation + keypoint export
├── 05_blender_to_sionna/            ← PLY export, cleaning, normalization
├── 06_rf_signal_generation/         ← Sionna CSI/CIR computation + spectrograms
├── 07_keypoint_linking/             ← Linking RF data with ground-truth keypoints
├── 08_live_hand_tracking/           ← MediaPipe → Blender live receiver
└── 09_reconstruction/               ← Rebuilding Blender animation from JSON
```

---

## Current Status (as of Jan 2025)

- [x] Hand rig modeled and rigged in Blender
- [x] Gesture animation scripts (open, fist, wave, thumbs up/down, random)
- [x] Per-frame PLY export from Blender
- [x] PLY cleaning and metric-scale normalization
- [x] Sionna scene setup (empty, floor-only, floor+walls)
- [x] CSI/CIR generation per frame
- [x] RF data saved as .npz
- [x] Ground-truth keypoints saved as JSON
- [x] Linked dataset (RF + keypoints) in .npz
- [x] Spectrogram generation
- [x] Live hand tracking via MediaPipe → Blender (v3)
- [ ] MediaPipe → Blender ML bridge (in progress)
- [ ] Gesture optimization model training (next step)

---

## Key Dependencies

```
Python 3.10
sionna-rt == 1.2.1
mitsuba == 3.7.1
drjit == 1.2.0
trimesh
numpy
matplotlib
imageio
mediapipe
opencv-python
bpy (Blender Python API)
```
