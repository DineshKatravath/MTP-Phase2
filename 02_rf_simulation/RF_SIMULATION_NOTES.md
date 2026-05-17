# RF Simulation — Problems and Solutions

This document is a companion to the main [README.md](README.md) in this module. It records the practical problems encountered while building the Sionna-based mmWave RF simulation stage, the fixes that worked, and the operational quirks that anyone re-running this stage needs to know about.

It is written as a working log rather than a polished tutorial, so future users can recognise the same failure modes quickly instead of re-discovering them.

---

## 1. Scene Geometry Problems

### 1.1 Empty-space simulation produced zero signal

The first version of the scene contained only the hand mesh in free space, with no surrounding geometry. The Sionna PathSolver returned essentially zero channel energy: `|H(f)|` was flat at the noise floor and the CIR had no detectable peaks.

**Cause.** With no walls, floor, or ceiling, the only scattering surface is the hand itself. At 28 GHz the hand's reflective cross-section is small, and with no multipath the line-of-sight path between Tx and Rx dominates everything else. There is no rich channel for the hand to perturb.

**Fix.** Built up the scene in three stages — empty → floor → full room enclosure — and validated each stage against the others. The final scene is `floor + 4 walls + ceiling` with reflection depth 6. This is the configuration described in `scripts/save_rf_parallel_v2.py` and `rf_scene.py` of the live demo.

### 1.2 Floor-only scene was LOS-dominated and featureless

Adding only a floor plane gave a non-zero CSI, but the channel was dominated by the smooth ground reflection. The CSI magnitude curve was a clean U-shape regardless of which gesture was placed in the scene — gesture discrimination was effectively impossible.

**Fix.** Promoted the scene to the full room enclosure. The walls and ceiling produce multiple early-delay multipath components that the hand can measurably perturb. This single change is what made gesture classification work at all.

### 1.3 Blender 5.x export was incompatible with Mitsuba 2.1

Initially the plan was to export the scene directly from Blender to Mitsuba XML using the Mitsuba-Blender add-on. Blender 5.x produced XML the Sionna-bundled Mitsuba 3 could not parse — material nodes, mesh references, and emitter syntax had all moved between Mitsuba versions.

**Fix.** Abandoned the Blender exporter route entirely. Instead, the scene is built programmatically in Python by `rf_scene.py`, which writes a minimal Mitsuba 3 XML string referencing the per-frame hand `.ply` and a fixed room geometry. This decouples the simulation from Blender's add-on version and lets the same scene template work across all live and offline runs.

### 1.4 Mesh referenced from XML but file path was wrong

The first programmatic scenes referenced the hand mesh with a relative path. Sionna resolves paths relative to its working directory, not to the XML file, so depending on where the script was launched the solver either silently used a stale mesh or failed to find any mesh at all.

**Fix.** All mesh references in the generated XML are absolute paths produced by `os.path.abspath()` inside `rf_scene.py`. There is now no ambiguity about which `.ply` is being simulated.

---

## 2. Material and Physics Problems

### 2.1 Default plastic material did not behave like skin at 28 GHz

The hand mesh was initially assigned Sionna's default `itu_plastic` material. At 28 GHz the absorption and reflection coefficients of plastic are very different from biological tissue, and the resulting CSI did not match published expectations.

**Fix.** Switched to the ITU-R P.2040 biological tissue parameters: relative permittivity `εr = 17.3`, conductivity `σ = 25.6 S/m`. These values produce physically plausible scattering and are the values quoted in the main README. The change is implemented in `csi_gen_predefined_skin.py` and carried into all production scripts.

### 2.2 1,000,000 rays per frame was too slow

Sionna's default ray count of 1M produced very clean CSI but made each frame take tens of seconds to simulate — unworkable for a 10,000-frame dataset and impossible for the live demo.

**Fix.** Reduced ray count to 50,000 (a 20× reduction) and verified that the per-frame CSI statistics at hand scale remain stable. The reduction does not measurably hurt gesture discriminability because the dominant multipath components are captured well below 50K rays. Production scripts use 50K rays.

### 2.3 Reflection depth too shallow lost diffuse contributions

Reflection depth 2 — Sionna's default — produced a CSI dominated by the first-order wall bounce. Higher-order contributions, which carry much of the hand's pose-dependent signal, were absent.

**Fix.** Reflection depth was raised to 6. This captures multi-bounce paths off the walls and back through the hand region. Increasing further to 8 did not visibly change the CSI and only added cost.

---

## 3. Tx / Rx Placement Problems

### 3.1 Hand not centred between Tx and Rx

Initially Tx and Rx were placed at fixed coordinates in the room. As the hand moved during animation (especially during wrist-rotation gestures), the hand drifted outside the Tx–Rx beam region and the CSI lost its sensitivity to pose.

**Fix.** Tx and Rx are now computed per-frame from the hand mesh's bounding-box centroid by `preprocess_frames.py` (in `01_blender_gesture_modeling/`). They are placed on opposite sides of the hand along the X axis, equidistant from the centroid. This guarantees the hand is always in the Tx–Rx gap regardless of its position in world coordinates.

### 3.2 Inconsistent hand scale across frames

Different exports from Blender ended up at different scales because of accumulated transform changes during animation. A 5 cm hand and a 50 cm hand produce very different CSI even at the same pose.

**Fix.** `preprocess_frames.py` normalises each mesh to a fixed physical size (in metres) before the scene is built. After this step every frame's hand has the same physical extent and only the pose varies between frames.

---

## 4. Parallel Simulation Problems

### 4.1 Out-of-order results in the parallel workers

The first parallel version (`save_rf_data_parallel.py`) used a simple `multiprocessing.Pool.imap_unordered` to dispatch frames. Workers finished frames out of order, so the saved `.npz` had rows in a permutation of the original frame indices. This broke the `link_rf_to_blender.py` alignment step downstream.

**Fix.** The production version (`save_rf_parallel_v2.py`) records the frame ID alongside the CSI in each worker's return, and the collector sorts results by frame ID before saving. The output `.npz` is now guaranteed to be in frame order.

### 4.2 TensorFlow GPU contention across workers

Each Sionna worker process imported TensorFlow at startup and tried to allocate the full GPU memory. With more than one worker, TF would fail with an out-of-memory error or hang.

**Fix.** Workers explicitly call `tf.config.experimental.set_memory_growth(gpu, True)` before any model is created, and the worker pool size is tuned to match available GPU memory (typically 2–4 workers on a 12 GB GPU). The configuration lives in `sionna_setup.py` and is imported by every worker before Sionna is touched.

### 4.3 Worker pool not shutting down cleanly on Ctrl+C

Killing the parallel script with `Ctrl+C` left orphaned worker processes that kept holding GPU memory until the machine was rebooted.

**Fix.** The script now wraps its main loop in a `try / finally` that calls `pool.terminate()` and `pool.join()`, and the live-demo `run_pipeline.py` orchestrator sends `SIGINT` to every child process explicitly on shutdown. Orphaned workers no longer occur.

### 4.4 Live-mode dispatching from a file-watch

For the live demo the simulation needs to start as soon as Blender writes a new `.ply` file. A naive polling loop missed frames when the disk write happened mid-poll.

**Fix.** `save_rf_parallel_v2.py` watches the frame directory with a 50 ms poll and only submits a `.ply` once its file size has stayed constant for one poll interval (a cheap proxy for "the write is finished"). This eliminates the half-written-file errors that the first version produced.

---

## 5. Output and Inspection Problems

### 5.1 `.npz` archives grew enormous

A 10,000-frame dataset at 256 subcarriers and a full CIR per frame produced multi-gigabyte `.npz` files. Loading the whole archive into memory in the ML stage was painful.

**Fix.** The archives are stored as compressed `.npz`, and `load_npz.py` exposes memory-mapped accessors so that the ML stage can stream frames instead of loading them all at once.

### 5.2 Hard to tell whether a result was "good"

Early in development there was no quick way to look at a single frame's CSI and judge whether the simulation had worked. The team kept re-running long batches just to inspect a few frames visually.

**Fix.** `load_npz.py` was extended into an inspection utility that prints summary statistics (mean magnitude, dynamic range) and can pop up a Matplotlib plot for any single frame. This made debugging much faster.

---

## 6. Environment Problems

### 6.1 Sionna version drift broke the PathSolver call signature

A point upgrade of Sionna changed the PathSolver's expected keyword arguments, and existing scripts started raising `TypeError` at runtime.

**Fix.** The Sionna version is pinned to `1.2.2` in `requirements.txt` and the local `sionna_env/` virtualenv is kept frozen. Anyone setting this up fresh should install `sionna==1.2.2` explicitly — do not allow pip to pick a newer version.

### 6.2 Mitsuba bundled with Sionna vs system Mitsuba

If the system already has a `mitsuba` Python package installed, Sionna sometimes loaded the system version instead of the bundled one, and scene loading failed with cryptic XML errors.

**Fix.** The environment is created with `--no-system-site-packages` so `sionna_env/` is fully isolated, and the bundled Mitsuba is the only one on the import path.

---

## 7. Summary of Recurring Lessons

- **Always validate the scene before tuning the physics.** Almost every "the CSI looks wrong" investigation traced back to scene geometry, not to ray counts or materials.
- **Pin the simulator version.** Sionna and Mitsuba evolve quickly; an unpinned environment guarantees future breakage.
- **Make Tx/Rx follow the hand, not the other way round.** Per-frame placement from the mesh centroid is the only reliable way to keep CSI sensitivity to pose across long animations.
- **Sort by frame ID in any parallel pipeline.** Out-of-order results are the single biggest source of downstream alignment bugs.
- **Use absolute paths in generated XML.** Sionna's working directory is not the same as the script's directory, and "the path was wrong" wastes hours.
