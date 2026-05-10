"""
sionna_live.py
==============
Watches hand_frames_live/ for new PLY files, runs the full pipeline:
  clean → normalize → Sionna RF → save .npz

Optimizations vs original:
  - Persistent scene (walls/floor/ceiling built once, hand swapped via scene.edit())
  - Background prefetch thread: normalizes PLY N+1 while Sionna solves PLY N
  - Reduced PathSolver cost: max_depth=2, samples_per_src=50_000,
    no diffraction, no refraction (physically justified at 28 GHz)
  - Complete room: floor + ceiling + 4 walls
  - No matplotlib — plotting handled by rf_plotter.py in main process
"""

import os
import time
import shutil
import threading
import queue
import numpy as np
import trimesh

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

from sionna.rt import (
    load_scene_from_string, Transmitter, Receiver,
    PlanarArray, PathSolver, RadioMaterial, SceneObject
)

# ------------------------------------------------------------------ CONFIG ---

BASE          = "/Users/dinesh/Documents/mtp/hand_models/live_rf_plot_hand"
LIVE_DIR      = os.path.join(BASE, "hand_frames_live")
CLEAN_DIR     = os.path.join(BASE, "hand_frames_live_clean")
NORM_DIR      = os.path.join(BASE, "hand_frames_live_normalized")
RF_DIR        = os.path.join(BASE, "rf_output")
PROCESSED_DIR = os.path.join(LIVE_DIR, "processed")

FREQUENCY       = 28e9
BANDWIDTH       = 400e6
NUM_SUBCARRIERS = 64
MAX_DEPTH       = 2          # reduced from 6: dominant paths at 28 GHz
SAMPLES_PER_SRC = 50_000     # reduced from 1M default: ~10x speedup

TARGET_SPAN_M   = 0.20
CLEARANCE_RATIO = 0.75
IDLE_TIMEOUT    = 30.0
POLL_INTERVAL   = 0.05

# Sliding window: how many normalized PLYs to keep ready ahead of solver.
# The background thread stays this many frames ahead of the Sionna solver.
PREFETCH_AHEAD  = 2

HAND_OBJECT_NAME = "hand"

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH / 2, BANDWIDTH / 2, NUM_SUBCARRIERS
)).astype(np.float32)

# --------------------------------------------------------- STATIC SCENE XML ---
# Complete room: floor + ceiling + 4 walls.
# Hand mesh is NOT in this XML — it is injected per-frame via scene.edit().
# Material IDs have no per-frame suffix since this XML is loaded only once.

STATIC_SCENE_XML = """<scene version="2.1.0">

    <bsdf type="itu-radio-material" id="mat-floor">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.1"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-ceiling">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.15"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-wall-front">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.2"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-wall-back">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.2"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-wall-left">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.2"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-wall-right">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.2"/>
    </bsdf>

    <!-- Floor -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="5" z="1"/>
            <translate x="0" y="0" z="-0.15"/>
        </transform>
        <ref id="mat-floor" name="bsdf"/>
    </shape>

    <!-- Ceiling -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="5" z="1"/>
            <translate x="0" y="0" z="1.5"/>
        </transform>
        <ref id="mat-ceiling" name="bsdf"/>
    </shape>

    <!-- Front wall (+Y) -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="1" y="0" z="0" angle="90"/>
            <translate x="0" y="0.5" z="0"/>
        </transform>
        <ref id="mat-wall-front" name="bsdf"/>
    </shape>

    <!-- Back wall (-Y) -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="1" y="0" z="0" angle="90"/>
            <translate x="0" y="-0.5" z="0"/>
        </transform>
        <ref id="mat-wall-back" name="bsdf"/>
    </shape>

    <!-- Left wall (-X) -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="0" y="1" z="0" angle="90"/>
            <translate x="-2.0" y="0" z="0"/>
        </transform>
        <ref id="mat-wall-left" name="bsdf"/>
    </shape>

    <!-- Right wall (+X) -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="0" y="1" z="0" angle="90"/>
            <translate x="2.0" y="0" z="0"/>
        </transform>
        <ref id="mat-wall-right" name="bsdf"/>
    </shape>

</scene>"""

# ---------------------------------------------------------- SCENE BUILD ------

def build_scene():
    """
    Load static room geometry once. TX/RX positions are set later
    after calibration — placeholders used here.
    Returns (scene, solver, skin_material).
    """
    scene = load_scene_from_string(STATIC_SCENE_XML)
    scene.frequency = FREQUENCY

    arr = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.tx_array = arr
    scene.rx_array = arr

    # Placeholder positions — overwritten after calibration
    scene.add(Transmitter(name="tx", position=[0.0, 0.0, 0.5]))
    scene.add(Receiver   (name="rx", position=[0.0, 0.0, 0.5]))

    # One skin material for all frames — registered once with the scene
    skin = RadioMaterial(
        name="skin",
        relative_permittivity=17.3,
        conductivity=25.6,
    )
    scene.add(skin)

    # PathSolver created once — reused every frame
    solver = PathSolver()

    return scene, solver, skin


# ---------------------------------------------------------- HAND MESH SWAP ---

def swap_hand(scene, norm_path, skin):
    """
    Replace the hand mesh in the persistent scene using scene.edit().
    Single call removes old + adds new → only one BVH rebuild.
    """
    new_hand = SceneObject(
        fname=norm_path,
        name=HAND_OBJECT_NAME,
        radio_material=skin,
    )
    if HAND_OBJECT_NAME in scene.objects:
        scene.edit(remove=HAND_OBJECT_NAME, add=new_hand)
    else:
        scene.edit(add=new_hand)


# ----------------------------------------------------------- CLEAN / NORM ----

def clean_frame(src, dst):
    mesh = trimesh.load(src, process=False)
    mesh.export(dst)


def compute_scale_factor(clean_path):
    mesh = trimesh.load(clean_path, process=False)
    verts = np.array(mesh.vertices)
    span  = verts.max() - verts.min()
    sf    = TARGET_SPAN_M / span
    print(f"[calib] span={span:.4f} units  scale={sf:.6f} m/unit")
    return sf


def normalize_frame(src, dst, sf):
    """Load, scale, export. Returns (centre, span) of scaled mesh."""
    mesh  = trimesh.load(src, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    mesh.vertices = verts * sf
    mesh.export(dst)
    scaled = verts * sf
    return scaled.mean(axis=0), scaled.max() - scaled.min()


def calibrate_tx_rx(norm_path):
    mesh   = trimesh.load(norm_path, process=False)
    verts  = np.array(mesh.vertices)
    centre = verts.mean(axis=0)
    span   = verts.max() - verts.min()
    clr    = span * CLEARANCE_RATIO
    tx = np.array([centre[0] - clr, centre[1], centre[2]])
    rx = np.array([centre[0] + clr, centre[1], centre[2]])
    print(f"[calib] centre={centre.round(4)}  span={span:.4f}m")
    print(f"[calib] TX={tx.round(4)}  RX={rx.round(4)}")
    print(f"[calib] separation={rx[0]-tx[0]:.4f}m  "
          f"hand fills ~{span/(rx[0]-tx[0])*100:.0f}%")
    return tx, rx


# ------------------------------------------------------------- RF SOLVE ------

def process_rf(scene, solver, norm_path, skin, tx, rx):
    """Swap hand mesh, update TX/RX, solve paths, return (H, CIR)."""
    swap_hand(scene, norm_path, skin)

    # Update antenna positions (cheap — no BVH rebuild)
    scene.transmitters["tx"].position = tx
    scene.receivers["rx"].position    = rx

    paths = solver(
        scene=scene,
        max_depth=MAX_DEPTH,
        samples_per_src=SAMPLES_PER_SRC,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,   # expensive; skip for real-time
        refraction=False,           # negligible at 28 GHz (penetration depth ~0.8mm)
        diffraction=False,          # weak at 28 GHz (finger width ~1.4λ); skip for speed
    )

    result = paths.cfr(
        frequencies=subcarrier_freqs,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy",
    )

    if isinstance(result, tuple):
        H_real = np.array(result[0]).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.array(result[1]).squeeze().flatten()[:NUM_SUBCARRIERS]
    else:
        H_real = np.real(result).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.imag(result).squeeze().flatten()[:NUM_SUBCARRIERS]

    H   = H_real + 1j * H_imag
    CIR = np.fft.ifft(H)
    return H, CIR


# --------------------------------------------------------------- SAVE RF -----

def save_rf(H, CIR, idx):
    path = os.path.join(RF_DIR, f"rf_frame_{idx:04d}.npz")
    np.savez(path,
        H_real        = np.real(H).astype(np.float32),
        H_imag        = np.imag(H).astype(np.float32),
        CIR_real      = np.real(CIR).astype(np.float32),
        CIR_imag      = np.imag(CIR).astype(np.float32),
        ply_frame_idx = np.int32(idx),
        freq_axis     = subcarrier_freqs,
    )
    return path


# ------------------------------------------------- BACKGROUND NORM THREAD ----

# Sentinel — tells the consumer the producer finished
_DONE = object()


def _norm_worker(task_queue: queue.Queue,
                 ready_queue: queue.Queue,
                 scale_factor_ref: list):
    """
    Background thread.
    Consumes (fname, clean_path, norm_path) tasks from task_queue.
    Normalizes the PLY and puts (fname, norm_path, centre, span)
    into ready_queue for the main thread to consume.

    scale_factor_ref is a one-element list so the main thread can
    write the calibrated scale factor into it after the first frame.
    """
    while True:
        item = task_queue.get()
        if item is _DONE:
            ready_queue.put(_DONE)
            break

        fname, clean_path, norm_path = item
        sf = scale_factor_ref[0]
        if sf is None:
            # Scale factor not yet calibrated — signal failure
            ready_queue.put((fname, None, None, None))
            continue

        try:
            centre, span = normalize_frame(clean_path, norm_path, sf)
            ready_queue.put((fname, norm_path, centre, span))
        except Exception as e:
            print(f"[norm error] {fname}: {e}")
            ready_queue.put((fname, None, None, None))


# -------------------------------------------------------------------- MAIN ---

def main():
    print("=" * 60)
    print("sionna_live.py  (optimized — persistent scene + prefetch)")
    print(f"Watching: {LIVE_DIR}")
    print(f"PathSolver: max_depth={MAX_DEPTH}  samples={SAMPLES_PER_SRC:,}")
    print("=" * 60)

    for d in [CLEAN_DIR, NORM_DIR, RF_DIR, PROCESSED_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)
    os.makedirs(LIVE_DIR, exist_ok=True)

    # ── Build persistent scene once ──────────────────────────────────────
    print("[init] Building static scene (room geometry)...")
    scene, solver, skin = build_scene()
    print("[init] Done.\n")

    # Shared mutable state between main thread and norm worker
    scale_factor_ref = [None]   # written by main after calibration
    TX_POSITION      = None
    RX_POSITION      = None
    calibrated       = False

    # ── Start background normalization thread ─────────────────────────────
    task_queue  = queue.Queue()
    ready_queue = queue.Queue()

    norm_thread = threading.Thread(
        target=_norm_worker,
        args=(task_queue, ready_queue, scale_factor_ref),
        daemon=True,
    )
    norm_thread.start()

    last_seen_t = None
    idx         = 0
    t_start     = time.time()

    # Sliding window state:
    # pending_clean holds frames that have been cleaned but whose
    # norm result hasn't been consumed yet (in case the ready_queue
    # has items from a previous poll cycle).
    pending_clean: dict[str, str] = {}   # fname → clean_path
    processed_files = set()

    try:
        while True:

    # ---------------- PROCESS READY FRAMES FIRST ----------------
            while True:
                try:
                    item = ready_queue.get_nowait()
                except queue.Empty:
                    break

                if item is _DONE:
                    break

                fname_r, norm_path_r, centre, span = item

                if norm_path_r is None:
                    pending_clean.pop(fname_r, None)
                    continue

                try:
                    t0 = time.time()
                    H, CIR = process_rf(
                        scene, solver, norm_path_r, skin,
                        TX_POSITION, RX_POSITION
                    )

                    rf_path = save_rf(H, CIR, idx)

                    print(f"[{idx:04d}] {fname_r} → {os.path.basename(rf_path)}")

                except Exception as e:
                    print(f"[sionna error] {fname_r}: {e}")

                candidate_orig = os.path.join(LIVE_DIR, fname_r)
                if os.path.exists(candidate_orig):
                    shutil.move(candidate_orig, os.path.join(PROCESSED_DIR, fname_r))

                pending_clean.pop(fname_r, None)
                processed_files.add(fname_r)
                idx += 1

            # ---------------- NOW READ NEW FRAMES ----------------
            files = sorted(f for f in os.listdir(LIVE_DIR) if f.endswith(".ply"))

            if files:
                fname = files[0]   # FIFO
                candidate = os.path.join(LIVE_DIR, fname)

                if fname in processed_files or fname in pending_clean:
                    time.sleep(POLL_INTERVAL)
                    continue

                clean_path = os.path.join(CLEAN_DIR, fname)
                norm_path  = os.path.join(NORM_DIR, fname)

                try:
                    clean_frame(candidate, clean_path)
                except Exception as e:
                    print(f"[clean error] {fname}: {e}")
                    os.rename(candidate, candidate + ".err")
                    continue

                if not calibrated:
                    try:
                        sf = compute_scale_factor(clean_path)
                        scale_factor_ref[0] = sf
                        normalize_frame(clean_path, norm_path, sf)
                        TX_POSITION, RX_POSITION = calibrate_tx_rx(norm_path)
                        calibrated = True

                        verts = np.array(trimesh.load(norm_path, process=False).vertices)
                        centre = verts.mean(axis=0)
                        span   = verts.max() - verts.min()

                        ready_queue.put((fname, norm_path, centre, span))
                        pending_clean[fname] = clean_path

                    except Exception as e:
                        print(f"[calibration error]: {e}")
                        os.rename(candidate, candidate + ".err")
                        continue
                else:
                    task_queue.put((fname, clean_path, norm_path))
                    pending_clean[fname] = clean_path

            else:
                time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped manually.")
    finally:
        # Shut down background thread cleanly
        try: task_queue.put_nowait(_DONE)
        except queue.Full: pass
        norm_thread.join(timeout=5.0)

    total = time.time() - t_start
    print(f"\nDone. Processed {idx} frames in {total:.1f}s"
          + (f" ({idx/total:.2f} fr/s avg)" if total > 0 else ""))


if __name__ == "__main__":
    main()