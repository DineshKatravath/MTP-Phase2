# 02 — RF Simulation

This module contains the Sionna-based mmWave RF simulation stage of the pipeline. It takes per-frame `.ply` hand mesh files, places them in a physically realistic indoor room scene, and computes the 28 GHz Channel State Information (CSI) and Channel Impulse Response (CIR) for each frame using GPU-accelerated ray tracing.

## Background

The central hypothesis of the project is that different hand poses produce measurably different CSI/CIR signatures. This module validates that hypothesis by physically simulating how each hand mesh scatters a 28 GHz signal in an indoor room, producing labelled (pose, CSI) pairs at scale without any real radio hardware.

## Simulation Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| RF simulator | Sionna v1.2.2 | Differentiable, GPU-accelerated, physically accurate |
| Ray-tracing backend | Mitsuba 3 | High-performance path tracing via Mitsuba XML scenes |
| Scene description | Programmatic Mitsuba XML | Overcomes Blender 5.x / Mitsuba 2.1 exporter incompatibility |
| Hand material | ITU-R P.2040 biological tissue | Physically accurate mmWave absorption and reflection |

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Carrier frequency | 28 GHz |
| Bandwidth | 400 MHz |
| Subcarriers | 256 |
| Ray count | 50,000 (reduced from default 1,000,000 for speed; retains physically meaningful statistics at hand scale) |
| Reflection depth | 6 |
| Hand material permittivity (εr) | 17.3 |
| Hand material conductivity (σ) | 25.6 S/m |
| Scene | Floor + 4 walls + ceiling (full room enclosure) |

The ray count was reduced 20× from the default (1M → 50K) after verifying that channel statistics at hand scale are preserved, giving a significant per-frame speedup without loss of discriminability.

## Tx/Rx Placement

`preprocess_frames.py` (in `01_blender_gesture_modeling/`) computes transmitter and receiver positions from each `.ply` mesh:

1. Load the hand mesh and compute its bounding box centroid.
2. Normalise scale so the hand occupies a physically realistic size in metres.
3. Place Tx and Rx on opposite sides of the hand along the X axis, equidistant from the centroid, so the hand is always centred in the Tx–Rx gap.

This ensures consistent geometry across all frames regardless of the hand's original coordinate system.

## Scene Evolution

Three scene configurations were evaluated in order of increasing complexity:

### Stage 1 — Empty Space (hand only)
- Only the hand mesh in free space; no walls, floor, or ceiling.
- Result: **zero or near-zero signal energy.** The hand mesh alone with no background scatterers and incorrect material assignment does not produce a detectable channel response. CSI and CIR are flat.

### Stage 2 — Floor Added
- Hand mesh + a horizontal floor plane.
- Result: **LOS-dominated, featureless CSI.** The floor introduces signal energy (smooth U-shaped CSI magnitude, single CIR peak), but the lack of surrounding geometry means the hand's contribution is overwhelmed by the direct LOS path. Gesture discrimination is difficult.

### Stage 3 — Floor + 4 Walls + Ceiling (final configuration)
- Full room enclosure with reflection depth 6.
- Result: **rich, frequency-selective CSI.** Multiple multipath components at early delay bins. The CSI structure varies measurably between hand poses, enabling reliable gesture discrimination.

**Key finding:** a multipath-rich indoor room enclosure is necessary to amplify the hand's scattering contribution relative to the LOS path. The room environment is not optional — it is what makes RF-based pose discrimination possible.

## Scripts

### `scripts/sionna_setup.py`
Environment setup and Sionna/TensorFlow GPU configuration.

### `scripts/save_rf_data.py`
Single-process RF data generation. Iterates over a directory of `.ply` frames, generates a Mitsuba XML scene for each frame, runs the Sionna PathSolver, and saves the resulting CSI and CIR arrays.

### `scripts/save_rf_data_parallel.py`
First parallelised version using Python `multiprocessing`. Spawns a worker pool where each worker processes one `.ply` frame independently, reducing batch generation time proportionally to available CPU cores.

### `scripts/save_rf_parallel_v2.py`
Refined parallel version with improved in-order result collection and file-watch dispatching to support near-real-time operation in live demo mode.

### `scripts/load_npz.py`
Utility to inspect and validate saved `.npz` CSI/CIR archives.

### `exploration_scripts/`

| Script | Scene Configuration |
|--------|---------------------|
| `csi_gen_empty_space.py` | Hand only, no room geometry |
| `csi_gen_floor.py` | Hand + floor plane only |
| `csi_gen_walls_floor.py` | Hand + floor + walls (full room) |
| `csi_gen_predefined_skin.py` | Hand with ITU-R biological tissue material parameters |

These scripts document the three-stage scene evolution and are retained for reproducibility of the scene comparison experiments reported in Chapter 4 of the project report.

## Directory Structure

```
02_rf_simulation/
├── scripts/
│   ├── sionna_setup.py           # GPU and environment setup
│   ├── save_rf_data.py           # Single-process RF generation
│   ├── save_rf_data_parallel.py  # Parallelised RF generation (v1)
│   ├── save_rf_parallel_v2.py    # Parallelised RF generation (v2, live-mode capable)
│   └── load_npz.py               # NPZ archive inspection utility
├── exploration_scripts/
│   ├── csi_gen_empty_space.py    # Stage 1: empty space
│   ├── csi_gen_floor.py          # Stage 2: floor only
│   ├── csi_gen_walls_floor.py    # Stage 3: full room
│   └── csi_gen_predefined_skin.py  # ITU-R material parameter study
├── scene_configs/
│   └── temp_scene.xml            # Reference Mitsuba XML scene template
├── data/
│   ├── mmwave_gesture_sample.npy  # Sample RF output for quick inspection
│   └── no_movement/
│       ├── csi_cir_output.mp4     # Video of CSI/CIR for static hand
│       └── hand_frames/           # Sample .ply frames used for this experiment
└── sionna_env/                    # Local Python environment (excluded from Git)
```

## Output Format

Each processed batch produces a compressed `.npz` archive with the following arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `H_mat` | `(N, 256)` complex | CSI (Channel Frequency Response) per frame |
| `CIR_mat` | `(N, D)` complex | CIR (Channel Impulse Response) per frame |
| `frame_ids` | `(N,)` int | Frame index matching `.ply` filename |

These arrays are passed to `03_hand_mesh_pipeline/` for alignment with pose labels.

## Running RF Simulation

```bash
# Activate the Sionna environment
source 02_rf_simulation/sionna_env/bin/activate

# Single-process (small batches or debugging)
python scripts/save_rf_data.py --ply_dir /path/to/cleaned_frames --out rf_output.npz

# Parallelised (large dataset generation)
python scripts/save_rf_parallel_v2.py --ply_dir /path/to/cleaned_frames --out rf_output.npz
```

## Dependencies

- Sionna v1.2.2
- TensorFlow (GPU recommended)
- Mitsuba 3 (installed as part of Sionna)
- NumPy, Matplotlib

Install via:
```bash
pip install sionna==1.2.2
```
