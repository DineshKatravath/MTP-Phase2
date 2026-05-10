# 05 — Live Demo

This module implements the real-time end-to-end demonstration of the full pipeline. A live webcam feed drives a Blender hand model via MediaPipe, each resulting mesh frame is fed to a parallel Sionna RF simulation worker pool, and the resulting CSI/CIR is plotted in real time — showing how the wireless channel changes as the user changes their hand pose.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Live Demo Pipeline                       │
│                                                                  │
│  Webcam                                                          │
│    │                                                             │
│    ▼                                                             │
│  sender.py  ──(TCP socket)──►  Blender receiver script          │
│  (MediaPipe)                   (runs inside Blender)            │
│                                        │                         │
│                                        │ .ply export per frame  │
│                                        ▼                         │
│                               hand_frames_live/                  │
│                                        │                         │
│                            file-watch dispatcher                 │
│                                        │                         │
│                                        ▼                         │
│                          sionna_live_parallel_rf.py              │
│                          (multiprocessing worker pool)           │
│                                        │                         │
│                                  CSI / CIR                       │
│                                        │                         │
│                                        ▼                         │
│                              live_rf_plot.py                     │
│                           (real-time CSI/CIR plot)              │
└─────────────────────────────────────────────────────────────────┘
```

## Three Concurrent Processes

The live demo runs three processes in parallel, orchestrated by `run_pipeline.py`:

1. **MediaPipe sender** (`sender.py`) — captures webcam frames, extracts 21 hand landmarks via MediaPipe HandLandmarker, serialises landmarks, and streams them over a TCP socket to the Blender receiver.

2. **Sionna RF worker pool** (`sionna_live_parallel_rf.py`) — watches `hand_frames_live/` for new `.ply` files, dispatches each new frame to a multiprocessing worker that generates a Mitsuba XML scene and runs the Sionna PathSolver, then collects results in order and passes CSI/CIR to the plotter.

3. **Live CSI plotter** (`live_rf_plot.py`) — displays a real-time updating plot of `|H(f)|` (CSI magnitude across 256 subcarriers) and `|h(τ)|` (CIR magnitude), giving the user immediate visual feedback on how the channel changes with their hand pose.

The **Blender receiver** (`blender_receiver/reciever_live_frames.py`) runs inside Blender's embedded Python interpreter and cannot be managed as an external subprocess. It must be started manually inside Blender before launching `run_pipeline.py`.

## Startup Order

1. Open the hand rig `.blend` file from `01_blender_gesture_modeling/assets/source2/`.
2. Load and run `blender_receiver/reciever_live_frames.py` from Blender's Scripting workspace. The script starts a TCP socket listener and a 10 ms timer callback that drives the armature and exports frames.
3. In a terminal, run `run_pipeline.py` — it starts the Sionna worker pool first, waits for it to initialise (3 s), then starts the MediaPipe sender, waits for the first `.ply` frame to appear, and finally launches the live plotter.

```bash
# Activate the Sionna environment
source /path/to/sionna_env/bin/activate

# Launch the full pipeline
python live_rf_plot_hand/run_pipeline.py
```

## Scripts

### `run_pipeline.py`
Orchestrator. Starts and monitors all three external processes. Uses `threading` to relay `stdout` from each process to the terminal with a `[tag]` prefix. Handles `SIGINT` cleanly by terminating all subprocesses.

### `sender.py`
MediaPipe landmark sender. Captures webcam frames with OpenCV, passes each to MediaPipe's HandLandmarker model, serialises the 21 normalised 3D landmark coordinates, and sends them over TCP to the Blender receiver.

### `blender_receiver/reciever_live_frames.py`
Blender-side TCP receiver. Runs inside Blender's embedded Python interpreter using a `bpy.app.timers` callback at 10 ms intervals. Each callback: receives landmark coordinates from the socket, computes bone rotations from the palm coordinate frame and per-finger curl angles, applies temporal smoothing to suppress jitter, drives the Blender armature, and triggers a `.ply` export.

### `sionna_live_parallel_rf.py`
Parallelised Sionna RF worker (production version). Maintains a `multiprocessing.Pool` of workers. A file-watch thread monitors `hand_frames_live/` for new `.ply` files and submits them to the pool. An in-order result collector ensures that CSI outputs are delivered to the plotter in frame order, avoiding out-of-order display artefacts.

### `sionna_live_fast_v1.py` / `sionna_live_fast_v2.py` / `sionna_live_fast_version.py`
Earlier iterations of the Sionna live worker with successive optimisations (ray count tuning, batch queuing, worker pre-warming). Retained for reference.

### `sionna_live.py`
Single-process Sionna live worker. Used for debugging the RF scene configuration and verifying CSI output format before switching to the parallelised version.

### `live_rf_plot.py`
Real-time CSI/CIR plotter. Uses Matplotlib's animation API to update the CSI magnitude and CIR plots at each new frame. Reads CSI results from a shared queue written by the Sionna worker pool.

### `config.py`
Shared configuration: paths to `hand_frames_live/`, TCP host/port, Sionna simulation parameters (28 GHz, 256 subcarriers, 50K rays), and plotter update interval.

### `frame_utils.py`
Utilities: load a `.ply` file, normalise mesh scale, compute Tx/Rx positions from the hand bounding box centroid.

### `rf_scene.py`
Mitsuba XML scene builder for the live demo. Generates the per-frame XML scene string with the hand mesh embedded in the full room enclosure (floor + 4 walls + ceiling), ready for the Sionna PathSolver.

## Directory Structure

```
05_live_demo/
└── live_rf_plot_hand/
    ├── run_pipeline.py              # Pipeline orchestrator
    ├── sender.py                    # MediaPipe landmark sender
    ├── config.py                    # Shared configuration
    ├── frame_utils.py               # PLY loading and scale normalisation
    ├── rf_scene.py                  # Mitsuba XML scene builder
    ├── live_rf_plot.py              # Real-time CSI/CIR plotter
    ├── sionna_live.py               # Single-process Sionna worker (debug)
    ├── sionna_live_fast_v1.py       # Parallelised worker (v1)
    ├── sionna_live_fast_v2.py       # Parallelised worker (v2)
    ├── sionna_live_fast_version.py  # Parallelised worker (intermediate)
    ├── sionna_live_parallel_rf.py   # Parallelised worker (production)
    └── blender_receiver/
        └── reciever_live_frames.py  # Blender-side TCP receiver + PLY exporter
```

## Python Environments

The live demo uses two separate Python environments:

| Process | Environment |
|---------|-------------|
| `sender.py` (MediaPipe) | `06_mediapipe_integration/mediaPipe/mediaPipeEnv/` |
| `sionna_live_parallel_rf.py` and `live_rf_plot.py` | `02_rf_simulation/sionna_env/` |
| Blender receiver | Blender's embedded Python |

`run_pipeline.py` references these environments by absolute path. Update the `SENDER_PYTHON` and `SIONNA_PYTHON` variables in `run_pipeline.py` to match your local paths before running.

## Key Results

- The live Blender hand accurately mirrors the user's physical hand across multiple gestures, with ~10 ms armature update latency.
- Sionna CSI updates at approximately 1–2 frames per second in live mode with 50K rays per frame on a GPU-equipped machine.
- The real-time CSI plot shows visually distinct frequency-selective profiles for different hand poses, confirming that the RF channel responds measurably to pose changes in real time.

## Dependencies

- Sionna v1.2.2 + TensorFlow (GPU recommended)
- MediaPipe (`mediapipe` package)
- OpenCV (`opencv-python`)
- Matplotlib, NumPy

```bash
# Sionna environment
pip install sionna==1.2.2 matplotlib numpy

# MediaPipe environment
pip install mediapipe opencv-python
```
