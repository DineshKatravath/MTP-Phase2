# 00 — Project Evolution

This folder documents how the project evolved from start to present, what was tried, what failed, what worked, and why each decision was made. Reading this before diving into any individual folder will give you the full context.

---

## How It Started (January 2025, Last Week)

The project began with a single task: **model a realistic human hand in Blender that can make natural movements**. There was no RF signal work at this point — just getting a hand that could open, close, wave, and move like a real hand.

The first decision was whether to model the hand from scratch or download one. A **low-poly hand mesh was downloaded from Sketchfab** (https://sketchfab.com/3d-models/low-poly-hand-3d-model-19c9ac5c369a468a95f081a3cc2ad4ac) and rigged manually in Blender with 20 bones: 3 per finger, 2 for the thumb, 5 palm bones, and 1 wrist bone.

---

## Stage 1 — Hand Modeling and Basic Movements

Once the rig existed, the next task was to control it with Python scripts inside Blender. The early scripts (`palm_fist_palm.py`, `gestures_with_anim.py`) used simple curl angles per finger, controlled by `rotation_euler` on each bone. The approach: define gesture keypoints manually (open = 0°, fist = 85–100°) and interpolate between them using Blender's animation system.

This worked well for predefined gestures. The `config.py` + `utils.py` + `gestures.py` modular structure was created to make it easy to call gestures like `gestures.fist()` or `gestures.open_hand()` from a single script.

**What worked:** Clean modular structure, good-looking gestures for predefined poses.  
**What was lacking:** No global hand orientation, no realistic random variation, no way to generate diverse training data automatically.

---

## Stage 2 — Global Orientation and Richer Animation

The next evolution was adding global hand orientation (rotating the whole armature object, not just the bones) and making the hand move in 3D space. This is when `gestures_anim_global.py` was written — it added wrist rotation in 3D and global translation, producing sequences where the hand moves forward, backward, and rotates between gestures.

The keypoint logging was also added here: after creating the animation, a second pass over all frames extracts each bone's world position and rotation, saving everything to `hand_motion_full.json`. This JSON became the ground truth file that would later be paired with RF data.

**Key insight:** Each frame of the animation = one "slow-time" snapshot for the spectrogram. The JSON records what the hand was doing at each frame, which would eventually be paired with the RF signal computed for that same frame's mesh.

---

## Stage 3 — 21-Point Parameterization

MediaPipe (the real-world hand tracking library) uses 21 keypoints to represent a hand. The question was: how do we represent our Blender hand in the same 21-point format so the two systems can talk to each other?

`gestures_anim_21rep.py` was written to extract exactly the same 21 joint positions from the Blender rig that MediaPipe would give for a real hand:
- wrist
- 5 palm bones (palm_thumb, palm_index, palm_middle, palm_ring, palm_pinky)
- 3 bones per finger = 12 joints
- 2 thumb joints

Both global coordinates and wrist-relative coordinates are saved, along with per-joint velocities. This was the bridge between Blender's bone-angle parameterization and MediaPipe's coordinate-based representation.

**Why this mattered:** Two possible approaches for the ML model — (a) use bone angles as input features, or (b) use joint world positions as input features. Having both formats available means either approach can be tested.

---

## Stage 4 — Random Pose Generation for Dataset Diversity

Predefined gestures (open, fist, wave, thumbs up) are only 5–6 classes. For a useful RF dataset, hundreds of diverse hand configurations are needed. `gestures_anim_global_random.py` and `random_pose_ML.py` were written to generate random but anatomically plausible poses by sampling joint angles within realistic limits.

Key design: the base (MCP) joint angle drives the middle (PIP) and tip (DIP) joints through dependency rules (`MID_FACTOR = 0.68`, `TIP_FACTOR = 0.6`) so fingers don't look unnatural. Palm cupping also influences mid and tip slightly.

`random_pose_ML.py` added a more structured approach for ML: render images of each random pose alongside the bone rotation data, with the idea of training a model that maps rendered images back to bone parameters.

---

## Stage 5 — Export to Sionna

With animation data being generated, the next step was exporting each frame's mesh as a PLY file for use in Sionna. `export_frames.py` exports the deformed mesh (with armature applied) for each frame in the animation as `hand_NNNN.ply`.

The PLY files then went through two preprocessing steps:
1. **Cleaning** (`clean_frames.py`): Strip any custom vertex attributes that cause Mitsuba to crash
2. **Normalization** (`preprocess_frames.py`): Scale the mesh from Blender units to real meters (hand span ≈ 0.20 m), without centering — preserving the natural position so TX/RX can be placed realistically

TX and RX positions were calculated by `preprocess_frames.py` itself: it measures the hand's natural center and span after scaling, then places TX and RX on either side with 75% of hand span as clearance.

**Critical lesson — the MTL/BSDF crash:** When Blender exports OBJ files, it includes a `.mtl` material file. Mitsuba reads this and creates a BSDF (visual material), but Sionna expects a RadioMaterial, not a BSDF. This caused `AttributeError: 'BSDF' object has no attribute 'add_object'`. The fix: use PLY format instead of OBJ (PLY carries no material info), and use `itu-radio-material` in the XML.

**Critical lesson — Mitsuba variant:** Sionna requires `llvm_ad_mono_polarized` (or `llvm_ad_rgb`) as the Mitsuba variant. The default `scalar_rgb` does not support radio materials. The exact versions required are `mitsuba==3.7.1`, `drjit==1.2.0`, `sionna-rt==1.2.1`.

---

## Stage 6 — Sionna Scene Evolution

Three scene configurations were tried, from simple to realistic:

| Script | Scene | Purpose |
|--------|-------|---------|
| `csi_gen_empty_space.py` | Hand mesh only | Baseline — pure hand reflection |
| `csi_gen_floor.py` | Hand + floor | Add ground-bounce multipath |
| `csi_gen_walls_floor.py` / `save_rf_data.py` | Hand + floor + 3 walls | Most realistic — current standard |

The floor+walls scene was settled on as the standard because it adds realistic multipath components that would also exist in a real indoor measurement environment.

The hand is assigned **skin EM properties** (εᵣ = 17.3, σ = 25.6 S/m at 28 GHz) via `RadioMaterial` in Python after loading the scene, because the XML only carries geometry — all radio physics are handled on the Python side.

---

## Stage 7 — RF Data Saving and Keypoint Linking

Once CSI/CIR generation worked, the data pipeline was completed:
- `save_rf_data.py`: processes every PLY frame → computes H (CSI) and CIR → saves `rf_frame_NNN.npz`
- `link_rf_to_blender.py`: scans all `rf_frame_NNN.npz` files → looks up the corresponding JSON keypoint entry (PLY index is 0-based, JSON frame is 1-based, so `json_frame = ply_idx + 1`) → saves one combined `linked_all_frames.npz` containing H_mat, CIR_mat, world_positions, local_positions, rotations_quat, joint_names

The final linked dataset shape: `(N_frames, 256)` for RF, `(N_frames, 20, 3)` for keypoints.

---

## Stage 8 — Spectrograms

`spectrogram_generator.py` builds the slow-time × frequency/delay spectrogram matrix. The key conceptual clarification here:
- **Delay axis** (CIR Y-axis): Δτ = 1/BW = 2.5 ns per tap — fixed by RF bandwidth, not frame rate
- **Slow-time axis** (frame axis): Δt = 1/FPS — set by Blender animation frame rate

The user inputs FPS at runtime so the time axis is always correctly labeled regardless of animation speed.

---

## Stage 9 — Live Hand Tracking

Parallel to the dataset pipeline, a live hand tracking system was built: MediaPipe detects real hand landmarks from a webcam → sends 21 joint coordinates over a socket → Blender receives them and drives the rig in real time.

Three receiver versions were written:
- `reciever.py`: basic curl from joint distances
- `reciever_v2.py`: added global hand orientation (wrist quaternion from palm normal)
- `reciever_v3.py`: best version — curl computed by projecting finger vectors onto palm plane (removes wrist-tilt artifacts), independent strength per finger, flip-stable quaternion slerp

**Current challenge:** The idea of using rendered Blender frames as input to MediaPipe (to create a training bridge between MediaPipe coordinates and Blender bone angles) ran into a problem — rendered images from Blender are not recognized by MediaPipe because the hand appearance doesn't match real skin texture. This is still being worked on (`random_pose_ML.py`).

---

## What's Next

1. **MediaPipe → Blender ML bridge**: Train a model on (MediaPipe 21-point coordinates, Blender bone angles) pairs from random poses, so live MediaPipe data can be cleanly converted to bone angles for the Blender rig
2. **Gesture parameterization for new synthesis**: Use the logged joint positions and angles to procedurally generate new unseen gestures by interpolation or latent-space sampling
3. **GT ↔ RF pairing for model training**: Use `linked_all_frames.npz` to train a model that maps RF signatures to gesture keypoints and eventually optimizes a gesture vocabulary for a specific person
