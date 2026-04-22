# Pipeline Overview

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Blender Hand Rig                                      │
│  Low-poly mesh + 20-bone armature (fingers, palms, wrist)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Gesture Animation                                     │
│  Python scripts drive bone rotations → keyframe animation       │
│  Gestures: open, fist, wave, thumbs up/down, random poses       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌─────────────────────────┐   ┌──────────────────────────────────┐
│  STAGE 3a: PLY Export   │   │  STAGE 3b: Keypoint Export       │
│  Per-frame deformed mesh│   │  JSON: world pos + rotations      │
│  hand_NNNN.ply          │   │  hand_motion_full.json            │
└────────────┬────────────┘   └──────────────────┬───────────────┘
             │                                   │
             ▼                                   │
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: PLY Preprocessing                                     │
│  clean_frames.py   → strip vertex attributes                    │
│  preprocess_frames.py → scale to meters, compute TX/RX pos      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: Sionna RF Simulation                                  │
│  28 GHz mmWave, 1 TX + 1 RX isotropic                          │
│  Scene: hand mesh + floor + 3 walls                             │
│  Propagation: LOS + specular/diffuse reflection + refraction    │
│              + diffraction, max_depth = 6                       │
│  Hand material: εᵣ = 17.3, σ = 25.6 S/m (skin at 28 GHz)      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 6: CSI / CIR Extraction                                  │
│  H(f) = CFR over 256 subcarriers, BW = 400 MHz                 │
│  h(τ) = IFFT(H)  →  CIR, Δτ = 2.5 ns/tap                     │
│  Saved as rf_frame_NNN.npz                                      │
└──────────────┬────────────────────────────────┬─────────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────────┐    ┌───────────────────────────────┐
│  STAGE 7: Keypoint Link  │    │  STAGE 8: Spectrograms        │
│  link_rf_to_blender.py   │    │  spectrogram_generator.py     │
│  H_mat (N,256) complex   │    │  slow-time × subcarrier/tap   │
│  CIR_mat (N,256) complex │    │  matrix  →  PNG + video       │
│  world_pos (N,20,3)      │    └───────────────────────────────┘
│  linked_all_frames.npz   │
└──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 9: [Future] Model Training                               │
│  Input: RF signature (H or CIR spectrogram)                     │
│  Output: gesture keypoints / gesture class                      │
│  Goal: find optimal gesture vocabulary per person               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Parallel Track — Live Hand Tracking

```
Webcam
  │
  ▼
MediaPipe (21 keypoints at ~30 fps)
  │
  ▼ (TCP socket, JSON per frame)
Blender Receiver Script (reciever_v3.py)
  │
  ├── Wrist quaternion from palm normal vectors
  └── Per-finger curl from plane-projected joint angles
  │
  ▼
Blender rig driven in real time (demo / validation)
```

---

## Data File Relationships

```
hand_NNNN.ply          ←── Blender export (export_frames.py)
       │
       ▼ (clean + scale)
hand_frames_normalized/hand_NNNN.ply
       │
       ▼ (Sionna processing)
rf_output/rf_frame_NNN.npz    hand_motion_full.json
       │                             │
       └──────────┬──────────────────┘
                  ▼ (link_rf_to_blender.py)
         linked/linked_all_frames.npz
           ├── H_mat          (N, 256) complex
           ├── CIR_mat        (N, 256) complex
           ├── world_positions(N, 20, 3)
           ├── local_positions(N, 20, 3)
           ├── rotations_quat (N, 20, 4)
           └── joint_names    (20,)
```

---

## Key Design Choices

| Decision | Choice | Reason |
|----------|--------|--------|
| Mesh format for Sionna | PLY | OBJ exports .mtl which causes BSDF crash in Sionna |
| Scene environment | Floor + 3 walls | Realistic multipath matching real indoor measurement |
| Hand material in XML | `itu-radio-material` (concrete placeholder) | XML-level placeholder; real skin properties assigned in Python |
| Hand EM in Python | εᵣ=17.3, σ=25.6 | Skin properties at 28 GHz |
| Mitsuba variant | `llvm_ad_mono_polarized` | Required for Sionna radio materials; scalar_rgb does not work |
| Keypoint format | World positions (21 joints) | Compatible with MediaPipe; easier to generalize |
| RF data format | .npz per frame, then combined | Easy to regenerate partial data; combined NPZ for ML training |
