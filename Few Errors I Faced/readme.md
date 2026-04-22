# Decisions and Lessons Learned

A record of every non-obvious choice made during the project, why it was made, and what was learned along the way.

---

## 1. PLY over OBJ for Sionna

**Problem:** Blender's OBJ exporter automatically creates a `.mtl` file with visual material info. Mitsuba (which Sionna uses internally) reads the `.mtl` and creates a BSDF (visual material object). Sionna then tries to call `radio_material.add_object()` on it — but a BSDF doesn't have that method → crash.

**Fix:** Export as PLY. PLY carries only geometry, no material references. The EM properties are then assigned in Python via `RadioMaterial`.

**Lesson:** For Sionna, geometry and radio physics must be completely separated. XML = geometry only. Python = all EM properties.

---

## 2. `itu-radio-material` in XML is a placeholder

**What it looks like:** The XML uses `<bsdf type="itu-radio-material">` with `value="concrete"`. This might seem wrong for a hand.

**Why:** Sionna's XML parser requires that every shape have a named material reference so it can register the object internally. The actual EM properties are overridden in Python right after loading via `obj.radio_material = RadioMaterial(...)`. So the XML material type doesn't matter — it just needs to exist and be named correctly.

**Lesson:** Always give unique IDs to materials per frame (`mat-hand-{idx}`) to avoid Sionna's internal material registry collision across frames.

---

## 3. Mitsuba Version Must Match Sionna Exactly

**Problem:** Installing Mitsuba separately (or with a different version) breaks Sionna because `radio_material` is a Sionna extension to Mitsuba, not a standard Mitsuba feature. Wrong versions give `encountered an unsupported XML element: <radio_material>` or BSDF errors.

**Required versions:**
```
mitsuba == 3.7.1
drjit   == 1.2.0
sionna-rt == 1.2.1
```

**Required variant:** `llvm_ad_mono_polarized` — must be set before importing sionna.rt.

---

## 4. PLY Cleaning Before Normalization

**Problem:** PLY files exported directly from Blender can contain custom vertex attributes (UV maps, vertex colors, shape key data) that cause Mitsuba's PLY loader to crash or produce incorrect geometry.

**Fix:** `clean_frames.py` reloads each PLY with trimesh and re-exports it, stripping all non-geometry data.

---

## 5. Scale Normalization — Don't Center

**Problem:** Blender uses arbitrary units. A hand in Blender might be 2 units wide, but in Sionna 1 unit = 1 meter, so the hand would appear 2 meters wide.

**Fix:** `preprocess_frames.py` measures the first frame's span and computes a scale factor so the hand is ~0.20 m wide (realistic hand span). It then scales all frames by this same factor.

**Important:** Only scale — do NOT center at origin. If you center each frame at origin, the hand's global motion (moving forward, backward) is lost and all frames look the same to the RF system. The hand's natural position in the scene must be preserved so TX/RX are correctly positioned relative to it.

**TX/RX placement:** Computed automatically as ±75% of hand span from the hand's natural center. For this dataset: TX = [-0.143, 0.020, -0.020], RX = [0.167, 0.020, -0.020].

---

## 6. Scene Environment Evolution

Three scene configurations were tried, in order of increasing realism:

**Empty scene** (`csi_gen_empty_space.py`)
- Only the hand mesh in free space
- Pure hand-reflected signals
- Good for isolating hand contribution
- Missing: ground bounce, wall reflections

**Floor only** (`csi_gen_floor.py`)
- Adds a 5×5 m floor at z = -0.5
- Adds one main ground-bounce multipath component
- Better than empty, still unrealistic

**Floor + walls** (`save_rf_data.py`, `csi_gen_walls_floor.py`, `spectrogram_generator.py`)
- Floor + front wall (y=0.5) + left wall (x=-2.0) + right wall (x=+2.0)
- Most realistic — matches a real indoor mmWave sensing setup
- **Current standard for all data collection**

---

## 7. The Slow-Time vs Delay-Time Confusion

**Problem:** Early code used the RF sampling rate to compute the inter-frame time gap, which is wrong. The sampling rate of the RF system determines delay resolution (CIR tap spacing), not how fast the hand moves between frames.

**Correct understanding:**
- **Delay axis (CIR):** Δτ = 1/BW = 1/400 MHz = 2.5 ns per tap. This is fixed by RF bandwidth, independent of animation.
- **Slow-time axis (frames):** Δt = 1/FPS. This is set by Blender's animation frame rate, entered by the user at runtime. It tells you how much time passed between one hand pose and the next.

These are two completely different time scales and must never be confused.

---

## 8. Finger Curl — Three Generations

**Version 1 (`reciever.py`):** Curl = `1 - (tip-to-base distance) / (sum of segment lengths)`. Simple, but doesn't account for hand orientation. When the wrist rotates, distances change even if fingers don't move.

**Version 2 (`reciever_v2.py`):** Same curl formula but with wrist quaternion computed from palm normal. Better global orientation. Curl still affected by wrist tilt.

**Version 3 (`reciever_v3.py`):** Curl computed by projecting finger bone vectors onto the palm plane before computing the angle between them. This removes wrist-tilt artifacts — curl only reflects actual finger bending. Separate strength and curl_max per finger (index needs tighter calibration to close fully). Quaternion slerp with dot-product flip detection for stability.

---

## 9. Keypoint Format — Angles vs Positions

**Two options considered:**

*Bone angles:* More compact (20 values for 20 bones), matches how the rig is controlled, requires mapping from MediaPipe coordinates to angles.

*Joint world positions:* 20 × 3 = 60 values, directly compatible with MediaPipe 21-point output, no intermediate conversion needed, easier to visualize.

**Decision:** Store both. The JSON (`hand_motion_full.json`) saves world positions, local positions, rotation quaternions, and velocities for every bone. The linked NPZ (`linked_all_frames.npz`) primarily uses world positions. Either can be used for model training.

---

## 10. The MediaPipe → Blender ML Bridge (In Progress)

**Goal:** Train a model that maps MediaPipe 21 joint coordinates → Blender bone angles, so live webcam hand tracking produces cleaner rig motion than direct coordinate mapping.

**Problem encountered:** Rendered Blender images are not recognized by MediaPipe because the synthetic hand doesn't have realistic skin texture. MediaPipe is trained on real photos and fails to detect the low-poly mesh.

**Status:** Still being explored. Options being considered: (a) use depth/skeleton overlay rendering instead of photorealistic; (b) use a purely coordinate-based model trained on (mediapipe coords, blender angles) pairs from random poses without needing images.
