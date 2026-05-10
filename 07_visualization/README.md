# 07 — Visualization

This module collects all static images and video recordings produced during the project, organised by pipeline stage. The media documents key results, demonstrates system behaviour, and illustrates the evolution of the scene environment.

## Images (`images/`)

### Blender and Rigging

| File | Description |
|------|-------------|
| `blender_rigged_hand.png` | Low-polygon hand mesh with completed 20-bone armature visible in Pose Mode. Octahedral glyphs show each bone in the hierarchy; the wrist root bone is at the base of the palm. |
| `blender_edit_mode_count.png` | Blender Edit Mode showing polygon/vertex count of the low-poly mesh, confirming mesh density suitable for real-time PLY export. |
| `blender_limit_rotation.png` | Blender's per-bone Limit Rotation constraint panel showing the biomechanically derived rotation limits applied to each bone. |

### Gesture Renders

Six canonical gestures rendered in Blender Pose Mode with the bone armature visible:

| File | Gesture |
|------|---------|
| `gesture_palm.png` | `OPEN` — fully extended open palm |
| `gesture_fist.png` | `FIST` — all fingers fully curled |
| `gesture_point.png` | `POINT` — index finger extended |
| `gesture_ok.png` | `PINCH` / OK — thumb and index touching |
| `gesture_thumbsup.png` | `THUMBS_UP` — thumb extended upward |
| `gesture_thumbsdown.png` | `THUMBS_DOWN` — thumb extended downward |

### Pose Parameterisation

| File | Description |
|------|-------------|
| `json_structure.png` | Screenshot of the per-frame JSON log format, showing the 32-DOF bone rotation vector and global wrist position for a single frame. |

### Tx/Rx Placement

| File | Description |
|------|-------------|
| `sionna_txrx_placement.png` | Top-down and 3D views of the hand mesh (wireframe) with Tx (red triangle) and Rx (green triangle) computed by `preprocess_frames.py`. Shows the hand centred in the Tx–Rx gap. |
| `placement_check.png` | Sanity check visualisation confirming scale normalisation — all coordinates in metres, scene origin at (0,0,0). |

### Sionna RF Scene

| File | Description |
|------|-------------|
| `scene_visualization.png` | 3D render of the full room enclosure scene (floor + 4 walls + ceiling) with the hand mesh, Tx, and Rx positioned inside. |

### CSI / CIR Plots

| File | Description |
|------|-------------|
| `sionna_csi_empty_space.png` | CSI and CIR for the empty-space configuration (hand only). Both show zero signal energy — confirms the hand mesh alone without room geometry produces no detectable channel response. |
| `sionna_csi_floor_only.png` | CSI and CIR with floor added. Smooth U-shaped CSI and single CIR peak indicate LOS dominance — featureless for gesture discrimination. |
| `sionna_csi_cir_plot.png` | CSI magnitude `|H(f)|` and CIR magnitude `|h(τ)|` for an open-palm pose in the full room configuration. Frequency-selective fading across 256 subcarriers and multiple early CIR multipath components. |
| `sionna_csi_live_fullroom.png` | CSI and CIR captured during the live MediaPipe demo in the full room configuration, showing rich multipath structure that varies between hand poses. |

### MediaPipe Live Demo

| File | Description |
|------|-------------|
| `mediapipe_landmarks.png` | MediaPipe 21-point hand skeleton overlaid on a live webcam frame. Wrist at base, palm anchors fanning to each finger, three knuckle landmarks per finger encode curl. |
| `mediapipe_demo_palm.png` | Open palm captured live (webcam window) mirrored by the Blender hand model (Top Orthographic view). |
| `mediapipe_demo_palm2.png` | Second open palm demonstration showing palm coordinate frame alignment. |
| `mediapipe_demo_thumbsup.png` | Thumbs-up gesture captured live and reproduced in Blender — four fingers correctly curled, thumb extended. |
| `mediapipe_terminal_connected.png` | Terminal screenshot showing the MediaPipe sender and Blender receiver successfully connected over TCP socket. |

## Videos (`videos/`)

| File | Description |
|------|-------------|
| `csi_cir_empty_output.mp4` | CSI and CIR animation for the empty-space scene configuration (hand only). Shows flat/zero channel across all frames. |
| `csi_cir_floor_output.mp4` | CSI and CIR animation with floor plane. Shows LOS-dominated, largely featureless channel. |
| `csi_cir_walls_floor_output.mp4` | CSI and CIR animation for the full room enclosure (floor + walls). Shows rich frequency-selective fading that varies measurably between hand poses. |
| `csi_cir_hand_28GHz.mp4` | CSI and CIR time-lapse at 28 GHz across all 8 gesture classes in the fixed-position dataset, demonstrating pose-dependent channel variation. |
| `csi_cir_output.mp4` | General CSI/CIR output video used for the project presentation. |
| `spectrogram_rolling.mp4` | Rolling spectrogram (time vs. subcarrier magnitude) across a gesture sequence, showing how the frequency-selective profile shifts between gesture classes over time. |

## Directory Structure

```
07_visualization/
├── images/
│   ├── blender_edit_mode_count.png
│   ├── blender_limit_rotation.png
│   ├── blender_rigged_hand.png
│   ├── gesture_fist.png
│   ├── gesture_ok.png
│   ├── gesture_palm.png
│   ├── gesture_point.png
│   ├── gesture_thumbsdown.png
│   ├── gesture_thumbsup.png
│   ├── json_structure.png
│   ├── mediapipe_demo_palm.png
│   ├── mediapipe_demo_palm2.png
│   ├── mediapipe_demo_thumbsup.png
│   ├── mediapipe_landmarks.png
│   ├── mediapipe_terminal_connected.png
│   ├── placement_check.png
│   ├── scene_visualization.png
│   ├── sionna_csi_cir_plot.png
│   ├── sionna_csi_empty_space.png
│   ├── sionna_csi_floor_only.png
│   ├── sionna_csi_live_fullroom.png
│   └── sionna_txrx_placement.png
└── videos/
    ├── csi_cir_empty_output.mp4
    ├── csi_cir_floor_output.mp4
    ├── csi_cir_hand_28GHz.mp4
    ├── csi_cir_output.mp4
    ├── csi_cir_walls_floor_output.mp4
    └── spectrogram_rolling.mp4
```
