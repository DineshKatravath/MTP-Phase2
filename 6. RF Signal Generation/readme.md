# 06 — RF Signal Generation

## What This Is

Scripts that load each PLY frame into Sionna's ray-tracing engine, compute the mmWave channel (CSI and CIR), and visualize or save the results.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `csi_gen_empty_space.py` | CSI/CIR for hand-only scene (no floor/walls) |
| `csi_gen_floor.py` | CSI/CIR with floor added |
| `csi_gen_walls_floor.py` | CSI/CIR with floor + 3 walls (visualization only) |
| `save_rf_data.py` | CSI/CIR with floor + 3 walls + saves .npz per frame |
| `spectrogram_generator.py` | Builds slow-time spectrogram from all frames |

**Current standard for data collection:** `save_rf_data.py` (floor + walls scene)

---

## RF System Parameters

```python
FREQUENCY       = 28e9          # 28 GHz mmWave
BANDWIDTH       = 400e6         # 400 MHz → delay resolution 2.5 ns
NUM_SUBCARRIERS = 256
MAX_DEPTH       = 6             # max ray bounce depth
TX_POSITION     = [-0.143, 0.020, -0.020]   # meters
RX_POSITION     = [ 0.167, 0.020, -0.020]   # meters
```

Both TX and RX are single-element isotropic antennas with vertical polarization:
```python
PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
```

---

## Propagation Modes Enabled

```python
paths = solver(
    scene=scene,
    max_depth=MAX_DEPTH,
    los=True,                   # Line-of-sight
    specular_reflection=True,   # Mirror-like reflections
    diffuse_reflection=True,    # Scattered reflections
    refraction=True,            # Through the hand mesh
    diffraction=True            # Around edges
)
```

All five modes are enabled to capture the full physical interaction between the mmWave signal and the hand.

---

## Hand EM Properties

The hand mesh is assigned skin electromagnetic properties at 28 GHz:

```python
skin = RadioMaterial(
    name="skin_{idx}",
    relative_permittivity=17.3,   # εᵣ for human skin at 28 GHz
    conductivity=25.6,            # σ in S/m
)
for obj in scene.objects.values():
    obj.radio_material = skin
```

These properties are applied in Python after loading the scene. The XML uses `itu-radio-material` as a placeholder only.

---

## CSI and CIR Computation

```python
result = paths.cfr(
    frequencies=subcarrier_freqs,
    num_time_steps=1,
    normalize_delays=True,
    normalize=False,
    out_type="numpy"
)
# result is a (real, imag) tuple
H   = H_real + 1j * H_imag      # Channel Frequency Response: (256,) complex
CIR = np.fft.ifft(H)            # Channel Impulse Response: (256,) complex
```

- **H(f):** Complex channel at each subcarrier frequency. `|H(f)|` shows how each frequency component is attenuated/amplified by the hand's presence.
- **h(τ):** IFFT of H(f). Shows the power arriving at each delay tap. Each tap corresponds to Δτ = 1/BW = 2.5 ns, representing a path of Δd = c × Δτ ≈ 0.75 m extra propagation distance.

---

## Scene XML Structure (Current Standard)

Three walls + floor added around the hand:

```
Floor:      5×5 m plane at z = -0.5 m
Front wall: 5×1 m at y = 0.5 m  (close to hand)
Left wall:  5×1 m at x = -2.0 m (further away)
Right wall: 5×1 m at x = +2.0 m (further away)
```

Each shape gets a unique material ID per frame (`mat-floor-{idx}`, etc.) to avoid Sionna's internal registry collision when processing multiple frames in one Python session.

---

## Spectrogram Generation

`spectrogram_generator.py` builds a 2D spectrogram matrix where:
- **Rows = frames (slow time):** Each row is one RF snapshot from one hand pose
- **Columns = subcarriers or delay taps (amplitude axis)**
- **Color = |H| or |h| in dB**

```
Slow-time spacing:  Δt = 1/FPS  (input by user at runtime)
Delay tap spacing:  Δτ = 1/BW = 2.5 ns  (fixed by bandwidth)
```

The script asks for Blender FPS, number of frames per spectrogram, and start frame index at runtime, then generates PNG + NPZ outputs and optionally a rolling-window video.

---

## Output Files

| File | Contents |
|------|---------|
| `rf_frame_NNN.npz` | H_real, H_imag, CIR_real, CIR_imag, ply_frame_idx, json_frame, freq_axis |
| `spectro_frames{start}-{end}.npz` | H_mat, CIR_mat, slow_time_axis, freq_axis, delay_axis_ns |
| `spectrogram_frames{start}-{end}.png` | CSI + CIR spectrogram images |
| `spectrogram_rolling.mp4` | Rolling-window spectrogram video |
| `csi_cir_*_output.mp4` | Frame-by-frame CSI/CIR magnitude video |

---

## Running

```bash
# Activate virtual environment
source venv/bin/activate

# Generate and save RF data for all frames
python save_rf_data.py

# Generate spectrogram
python spectrogram_generator.py
# → Enter FPS, number of frames, start frame when prompted
```

---

## Mitsuba Setup (Required Before Any Sionna Script)

```python
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")   # MUST be first, before sionna import
```

Required package versions:
```
mitsuba==3.7.1
drjit==1.2.0
sionna-rt==1.2.1
```
