"""
sionna_live_parallel.py
=======================
Compute-parallel RF pipeline:

    Blender → hand_frames_live/
                   ↓
              Dispatcher
                   ↓  task_queue (multiprocessing.Queue)
         ┌─────────┬─────────┬─────────┐
         │Worker 1 │Worker 2 │Worker N │   ← each owns its own
         │(Mitsuba)│(Mitsuba)│(Mitsuba)│     Mitsuba ctx + scene
         └─────────┴─────────┴─────────┘
                   ↓  result_queue
              Collector
                   ↓
              rf_output/  (.npz files)

Key design decisions
─────────────────────
• Each worker initialises Mitsuba and builds a persistent Sionna scene in
  its own subprocess — no shared GPU/LLVM state, no GIL bottleneck.

• Calibration (scale factor + TX/RX positions) is computed by the Dispatcher
  on frame 0, then broadcast to workers via a multiprocessing.Manager dict.
  Workers spin-wait (< 1 s) until calib_state['ready'] is True.

• The Collector reorders out-of-order results using a heap so that .npz files
  are always written in frame-index order, regardless of which worker finishes
  first.

• Workers keep the Sionna scene alive across frames.  Only the hand mesh is
  swapped (scene.edit) — one BVH rebuild per frame, not a full scene reload.

• A background normalisation thread inside each worker prefetches the PLY
  normalisation while Sionna is solving, matching the pattern from the
  single-process version.

Usage
──────
    python sionna_live_parallel.py [--workers N]

    N defaults to multiprocessing.cpu_count() // 2 (leave half for the OS /
    Blender).  For a 10-core machine, 4-5 workers is a good starting point.
"""

import os
import sys
import time
import shutil
import queue
import heapq
import argparse
import threading
import multiprocessing as mp
from multiprocessing.managers import SyncManager

import numpy as np
import trimesh

# -------------- CONFIG ----------------------

BASE          = "/Users/dinesh/Documents/mtp/hand_models/live_rf_plot_hand"
LIVE_DIR      = os.path.join(BASE, "hand_frames_live")
CLEAN_DIR     = os.path.join(BASE, "hand_frames_live_clean")
NORM_DIR      = os.path.join(BASE, "hand_frames_live_normalized")
RF_DIR        = os.path.join(BASE, "rf_output")
PROCESSED_DIR = os.path.join(LIVE_DIR, "processed")

FREQUENCY       = 28e9
BANDWIDTH       = 400e6
NUM_SUBCARRIERS = 64
MAX_DEPTH       = 6
SAMPLES_PER_SRC = 50_000

TARGET_SPAN_M   = 0.20
CLEARANCE_RATIO = 0.75
IDLE_TIMEOUT    = 30.0
POLL_INTERVAL   = 0.05

HAND_OBJECT_NAME = "hand"

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH / 2, BANDWIDTH / 2, NUM_SUBCARRIERS
)).astype(np.float32)

# --------------------------------------------------------- STATIC SCENE XML ---

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

    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="5" z="1"/>
            <translate x="0" y="0" z="-0.15"/>
        </transform>
        <ref id="mat-floor" name="bsdf"/>
    </shape>
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="5" z="1"/>
            <translate x="0" y="0" z="1.5"/>
        </transform>
        <ref id="mat-ceiling" name="bsdf"/>
    </shape>
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="1" y="0" z="0" angle="90"/>
            <translate x="0" y="0.5" z="0"/>
        </transform>
        <ref id="mat-wall-front" name="bsdf"/>
    </shape>
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="1" y="0" z="0" angle="90"/>
            <translate x="0" y="-0.5" z="0"/>
        </transform>
        <ref id="mat-wall-back" name="bsdf"/>
    </shape>
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="0" y="1" z="0" angle="90"/>
            <translate x="-2.0" y="0" z="0"/>
        </transform>
        <ref id="mat-wall-left" name="bsdf"/>
    </shape>
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="0" y="1" z="0" angle="90"/>
            <translate x="2.0" y="0" z="0"/>
        </transform>
        <ref id="mat-wall-right" name="bsdf"/>
    </shape>

</scene>"""

# ─────────────────────────────────────────────── SHARED GEOMETRY HELPERS ────

def clean_frame(src: str, dst: str):
    mesh = trimesh.load(src, process=False)
    mesh.export(dst)


def compute_scale_factor(clean_path: str) -> float:
    mesh = trimesh.load(clean_path, process=False)
    verts = np.array(mesh.vertices)
    span  = verts.max() - verts.min()
    sf    = TARGET_SPAN_M / span
    print(f"[calib] span={span:.4f}  scale={sf:.6f} m/unit", flush=True)
    return float(sf)


def normalize_frame(src: str, dst: str, sf: float):
    """Scale PLY by sf and save.  Returns (centre, span) of scaled mesh."""
    mesh  = trimesh.load(src, process=False)
    verts = np.array(mesh.vertices, dtype=np.float64)
    mesh.vertices = verts * sf
    mesh.export(dst)
    scaled = verts * sf
    return scaled.mean(axis=0), scaled.max() - scaled.min()


def calibrate_tx_rx(norm_path: str):
    mesh   = trimesh.load(norm_path, process=False)
    verts  = np.array(mesh.vertices)
    centre = verts.mean(axis=0)
    span   = verts.max() - verts.min()
    clr    = span * CLEARANCE_RATIO
    tx = np.array([centre[0] - clr, centre[1], centre[2]])
    rx = np.array([centre[0] + clr, centre[1], centre[2]])
    print(f"[calib] TX={tx.round(4)}  RX={rx.round(4)}", flush=True)
    return tx.tolist(), rx.tolist()


# ─────────────────────────────────────────────────────────── WORKER LOGIC ────

def _worker_init():
    """Called once at the start of each worker process."""
    # Mitsuba must be set before any sionna import
    import mitsuba as mi
    mi.set_variant("llvm_ad_mono_polarized")


def _build_worker_scene():
    """Build a persistent Sionna scene in this worker process."""
    from sionna.rt import (
        load_scene_from_string, Transmitter, Receiver,
        PlanarArray, PathSolver, RadioMaterial,
    )
    scene = load_scene_from_string(STATIC_SCENE_XML)
    scene.frequency = FREQUENCY

    arr = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.tx_array = arr
    scene.rx_array = arr

    scene.add(Transmitter(name="tx", position=[0.0, 0.0, 0.5]))
    scene.add(Receiver   (name="rx", position=[0.0, 0.0, 0.5]))

    skin = RadioMaterial(
        name="skin",
        relative_permittivity=17.3,
        conductivity=25.6,
    )
    scene.add(skin)
    solver = PathSolver()
    return scene, solver, skin


def _swap_hand(scene, norm_path: str, skin):
    from sionna.rt import SceneObject
    new_hand = SceneObject(
        fname=norm_path,
        name=HAND_OBJECT_NAME,
        radio_material=skin,
    )
    if HAND_OBJECT_NAME in scene.objects:
        scene.edit(remove=HAND_OBJECT_NAME, add=new_hand)
    else:
        scene.edit(add=new_hand)


def _solve_rf(scene, solver, norm_path: str, skin, tx, rx):
    _swap_hand(scene, norm_path, skin)
    scene.transmitters["tx"].position = tx
    scene.receivers["rx"].position    = rx

    paths = solver(
        scene=scene,
        max_depth=MAX_DEPTH,
        samples_per_src=SAMPLES_PER_SRC,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=False,
        diffraction=False,
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


def worker_process(worker_id: int,
                   task_queue: mp.Queue,
                   result_queue: mp.Queue,
                   calib_state: dict):
    """
    Long-lived worker process.

    Receives (frame_idx, fname, norm_path) tuples from task_queue.
    Waits until calib_state['ready'] is True (set by Dispatcher after
    calibrating on frame 0).
    Puts (frame_idx, fname, H, CIR) into result_queue.
    Exits cleanly on sentinel None.
    """
    print(f"[worker {worker_id}] starting", flush=True)

    # Initialise Mitsuba and build scene — done ONCE per worker
    _worker_init()
    scene, solver, skin = _build_worker_scene()
    print(f"[worker {worker_id}] scene ready", flush=True)

    # Background normalisation thread per worker
    # (normalise PLY N+1 while Sionna solves PLY N)
    local_norm_q:  "queue.Queue[object]" = queue.Queue()
    local_ready_q: "queue.Queue[object]" = queue.Queue()
    _NORM_DONE = object()

    def _norm_thread():
        while True:
            item = local_norm_q.get()
            if item is _NORM_DONE:
                local_ready_q.put(_NORM_DONE)
                break
            idx_n, fname_n, clean_n, norm_n, sf = item
            try:
                normalize_frame(clean_n, norm_n, sf)
                local_ready_q.put((idx_n, fname_n, norm_n, None))
            except Exception as e:
                local_ready_q.put((idx_n, fname_n, None, str(e)))

    nt = threading.Thread(target=_norm_thread, daemon=True)
    nt.start()

    # --- wait for calibration ---
    while not calib_state.get("ready", False):
        time.sleep(0.02)

    tx = calib_state["tx"]
    rx = calib_state["rx"]
    sf = calib_state["sf"]
    print(f"[worker {worker_id}] calib received  TX={tx}  RX={rx}", flush=True)

    pending: dict[str, tuple] = {}  # fname → (idx, norm_path)

    try:
        while True:
            # --- drain ready normalised frames first ---
            while True:
                try:
                    ritem = local_ready_q.get_nowait()
                except queue.Empty:
                    break
                if ritem is _NORM_DONE:
                    return
                idx_r, fname_r, norm_r, err = ritem
                if err or norm_r is None:
                    print(f"[worker {worker_id}] norm error {fname_r}: {err}", flush=True)
                    pending.pop(fname_r, None)
                    result_queue.put((idx_r, fname_r, None, None))
                    continue
                try:
                    H, CIR = _solve_rf(scene, solver, norm_r, skin, tx, rx)
                    result_queue.put((idx_r, fname_r, H, CIR))
                    print(f"[worker {worker_id}] solved frame {idx_r:04d}", flush=True)
                except Exception as e:
                    print(f"[worker {worker_id}] sionna error frame {idx_r}: {e}", flush=True)
                    result_queue.put((idx_r, fname_r, None, None))
                pending.pop(fname_r, None)

            # --- fetch new task ---
            try:
                task = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:  # sentinel
                local_norm_q.put(_NORM_DONE)
                break

            idx_t, fname_t, clean_t, norm_t = task

            # If norm_path already exists (calib frame pre-normalised by
            # Dispatcher), skip normalisation and solve directly.
            if os.path.exists(norm_t):
                try:
                    H, CIR = _solve_rf(scene, solver, norm_t, skin, tx, rx)
                    result_queue.put((idx_t, fname_t, H, CIR))
                    print(f"[worker {worker_id}] solved frame {idx_t:04d} (pre-norm)", flush=True)
                except Exception as e:
                    print(f"[worker {worker_id}] sionna error frame {idx_t}: {e}", flush=True)
                    result_queue.put((idx_t, fname_t, None, None))
            else:
                pending[fname_t] = (idx_t, norm_t)
                local_norm_q.put((idx_t, fname_t, clean_t, norm_t, sf))

    except KeyboardInterrupt:
        pass
    finally:
        nt.join(timeout=3.0)
        print(f"[worker {worker_id}] exited", flush=True)


# ───────────────────────────────────────────────────────── COLLECTOR LOGIC ───

def collector_process(result_queue: mp.Queue,
                      num_workers: int,
                      total_frames_event,
                      total_frames_ref: list):
    """
    Collects results from workers, reorders by frame_idx using a min-heap,
    writes .npz files in order.

    Exits when it has received sentinel None from all workers.
    """
    print("[collector] started", flush=True)
    heap   = []   # (frame_idx, fname, H, CIR)
    done_workers = 0
    next_to_write = 0
    written = 0
    buf: dict[int, tuple] = {}

    def _write(idx, fname, H, CIR):
        nonlocal written
        if H is None:
            print(f"[collector] skip frame {idx:04d} (error)", flush=True)
        else:
            path = os.path.join(RF_DIR, f"rf_frame_{idx:04d}.npz")
            np.savez(path,
                H_real        = np.real(H).astype(np.float32),
                H_imag        = np.imag(H).astype(np.float32),
                CIR_real      = np.real(CIR).astype(np.float32),
                CIR_imag      = np.imag(CIR).astype(np.float32),
                ply_frame_idx = np.int32(idx),
                freq_axis     = subcarrier_freqs,
            )
            print(f"[collector] wrote {os.path.basename(path)}", flush=True)
        written += 1

    try:
        while done_workers < num_workers:
            try:
                item = result_queue.get(timeout=0.5)
            except:
                continue

            if item is None:
                done_workers += 1
                continue

            idx_r, fname_r, H_r, CIR_r = item

            # Move original PLY to processed/
            candidate = os.path.join(LIVE_DIR, fname_r)
            if os.path.exists(candidate):
                shutil.move(candidate, os.path.join(PROCESSED_DIR, fname_r))

            buf[idx_r] = (fname_r, H_r, CIR_r)

            # Flush consecutive frames
            while next_to_write in buf:
                fn, H, CIR = buf.pop(next_to_write)
                _write(next_to_write, fn, H, CIR)
                next_to_write += 1

        # Flush remainder (gaps due to errors)
        for idx_r in sorted(buf):
            fn, H, CIR = buf[idx_r]
            _write(idx_r, fn, H, CIR)

    except KeyboardInterrupt:
        pass

    print(f"[collector] done. wrote {written} frames.", flush=True)


# ────────────────────────────────────────────────────────── DISPATCHER LOGIC ──

def dispatcher(num_workers: int, calib_state, task_queue: mp.Queue):
    """
    Main dispatcher: watches LIVE_DIR, cleans PLY files, sends tasks to
    workers.  Calibrates on frame 0 and populates calib_state.
    """
    print(f"[dispatcher] watching {LIVE_DIR}", flush=True)

    processed_files = set()
    idx             = 0
    sf              = None
    t_last          = time.time()

    while True:
        files = sorted(f for f in os.listdir(LIVE_DIR) if f.endswith(".ply"))

        if not files:
            if time.time() - t_last > IDLE_TIMEOUT:
                print("[dispatcher] idle timeout — sending sentinels", flush=True)
                break
            time.sleep(POLL_INTERVAL)
            continue

        t_last = time.time()

        for fname in files:
            if fname in processed_files:
                continue

            src        = os.path.join(LIVE_DIR, fname)
            clean_path = os.path.join(CLEAN_DIR, fname)
            norm_path  = os.path.join(NORM_DIR, fname)

            # Clean step (fast — just re-export)
            try:
                clean_frame(src, clean_path)
            except Exception as e:
                print(f"[dispatcher] clean error {fname}: {e}", flush=True)
                os.rename(src, src + ".err")
                processed_files.add(fname)
                continue

            # Calibration on very first frame
            if sf is None:
                try:
                    sf = compute_scale_factor(clean_path)
                    normalize_frame(clean_path, norm_path, sf)
                    tx, rx = calibrate_tx_rx(norm_path)

                    # Broadcast to all workers via Manager dict
                    calib_state["sf"]    = sf
                    calib_state["tx"]    = tx
                    calib_state["rx"]    = rx
                    calib_state["ready"] = True
                    print("[dispatcher] calibration broadcast done", flush=True)

                    # Send calib frame as a pre-normed task
                    task_queue.put((idx, fname, clean_path, norm_path))
                    processed_files.add(fname)
                    idx += 1
                except Exception as e:
                    print(f"[dispatcher] calibration error: {e}", flush=True)
                    os.rename(src, src + ".err")
                    processed_files.add(fname)
                continue

            # Normal frames
            task_queue.put((idx, fname, clean_path, norm_path))
            processed_files.add(fname)
            idx += 1

        time.sleep(POLL_INTERVAL)

    # Send sentinel to each worker
    for _ in range(num_workers):
        task_queue.put(None)

    print(f"[dispatcher] sent {idx} frames total", flush=True)


# MAIN

def main():
    parser = argparse.ArgumentParser(description="Parallel Sionna live pipeline")
    parser.add_argument(
        "--workers", "-n",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of parallel Sionna worker processes",
    )
    args = parser.parse_args()
    N = args.workers

    print("=" * 60)
    print(f"sionna_live_parallel.py  ({N} workers)")
    print(f"PathSolver: max_depth={MAX_DEPTH}  samples={SAMPLES_PER_SRC:,}")
    print("=" * 60)

    # Re-create output dirs
    for d in [CLEAN_DIR, NORM_DIR, RF_DIR, PROCESSED_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)
    os.makedirs(LIVE_DIR, exist_ok=True)

    # Shared calibration state (Manager dict is safe across processes)
    manager = SyncManager()
    manager.start()
    calib_state = manager.dict({"ready": False, "sf": None, "tx": None, "rx": None})

    # IPC queues
    # maxsize prevents the Dispatcher from flooding RAM with thousands of tasks
    task_queue   = mp.Queue(maxsize=N * 4)
    result_queue = mp.Queue()

    # Start workers
    workers = []
    for wid in range(N):
        p = mp.Process(
            target=worker_process,
            args=(wid, task_queue, result_queue, calib_state),
            daemon=True,
            name=f"worker-{wid}",
        )
        p.start()
        workers.append(p)
        print(f"[main] started worker {wid}  pid={p.pid}", flush=True)

    # Start collector
    col = mp.Process(
        target=collector_process,
        args=(result_queue, N, None, None),
        daemon=True,
        name="collector",
    )
    col.start()
    print(f"[main] started collector pid={col.pid}", flush=True)

    # Run dispatcher in the main process so KeyboardInterrupt works cleanly
    t0 = time.time()
    try:
        dispatcher(N, calib_state, task_queue)
    except KeyboardInterrupt:
        print("\n[main] interrupted — shutting down", flush=True)
        for _ in range(N):
            task_queue.put(None)

    # Wait for workers
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            print(f"[main] force-killing {p.name}", flush=True)
            p.kill()

    # Signal collector: all workers done
    result_queue.put(None)   # collector counts N sentinels from workers
    col.join(timeout=30)

    manager.shutdown()

    elapsed = time.time() - t0
    print(f"\n[main] pipeline finished in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # required for Mitsuba on macOS
    main()