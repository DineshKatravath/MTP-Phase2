import os
import numpy as np
import mitsuba as mi
import random
import shutil
import threading
import queue
import time
import matplotlib.pyplot as plt
import imageio
import trimesh

from matplotlib.backends.backend_agg import FigureCanvasAgg

from sionna.rt import (
    load_scene_from_string, Transmitter, Receiver,
    PlanarArray, PathSolver, RadioMaterial, SceneObject
)

mi.set_variant("llvm_ad_mono_polarized")

# =============================
# CONFIG
# =============================

FRAMES_DIR    = "/Users/dinesh/Documents/mtp/hand_models/with_wrist_movement/hand_frames_normalized"
RF_OUTPUT_DIR = "/Users/dinesh/Documents/mtp/hand_models/with_wrist_movement/rf_output_parallel_v3"

# Temp dir for normalized PLYs produced by the prefetch thread.
# Kept separate so we can clean up independently.
PREFETCH_DIR  = "/tmp/sionna_prefetch"

FREQUENCY       = 28e9
BANDWIDTH       = 400e6
NUM_SUBCARRIERS = 256
MAX_DEPTH       = 6

TX_POSITION = np.array([-0.113, -0.006, -0.002])
RX_POSITION = np.array([ 0.208, -0.006, -0.002])

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH / 2, BANDWIDTH / 2, NUM_SUBCARRIERS
)).astype(np.float32)

HAND_OBJECT_NAME = "hand"

# How many frames the prefetch thread stays ahead of the solver.
# 2-3 is enough; larger values just waste disk I/O.
PREFETCH_QUEUE_SIZE = 3

# =============================
# STATIC SCENE XML
# =============================

STATIC_SCENE_XML = """<scene version="2.1.0">

    <bsdf type="itu-radio-material" id="mat-floor">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.1"/>
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
    <bsdf type="itu-radio-material" id="mat-ceiling">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.15"/>
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
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="5" z="1"/>
            <translate x="0" y="0" z="1.5"/>
        </transform>
        <ref id="mat-ceiling" name="bsdf"/>
    </shape>

</scene>"""

# =============================
# PREFETCH THREAD
# Loads + normalizes each PLY on a background thread,
# writes a ready-to-use PLY to PREFETCH_DIR, and puts
# the output path into a queue for the main thread.
#
# The main thread (Sionna solver) and this thread never
# touch the same file simultaneously.
# =============================

# Sentinel value — tells the consumer the producer is done
_DONE = object()


def _prefetch_worker(src_paths: list[str], out_queue: queue.Queue):
    """
    Background thread: for each source PLY, load with trimesh,
    export a clean copy to PREFETCH_DIR, then enqueue the path.
    If a frame fails, enqueue None so the consumer can skip it.
    """
    os.makedirs(PREFETCH_DIR, exist_ok=True)

    for i, src in enumerate(src_paths):
        try:
            mesh = trimesh.load(src, process=False)
            dst  = os.path.join(PREFETCH_DIR, f"pre_{i:05d}.ply")
            mesh.export(dst)
            out_queue.put((i, dst))
        except Exception as e:
            print(f"[prefetch] frame {i} failed: {e}")
            out_queue.put((i, None))   # consumer will skip

    out_queue.put(_DONE)


# =============================
# SCENE SETUP — once
# =============================

def build_scene():
    scene = load_scene_from_string(STATIC_SCENE_XML)
    scene.frequency = FREQUENCY

    arr = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.tx_array = arr
    scene.rx_array = arr

    scene.add(Transmitter(name="tx", position=TX_POSITION.tolist()))
    scene.add(Receiver   (name="rx", position=RX_POSITION.tolist()))

    skin = RadioMaterial(
        name="skin",
        relative_permittivity=17.3,
        conductivity=25.6,
    )
    scene.add(skin)

    solver = PathSolver()
    return scene, solver, skin


# =============================
# HAND SWAP — per frame
# =============================

def swap_hand(scene, mesh_path, skin):
    new_hand = SceneObject(
        fname=mesh_path,
        name=HAND_OBJECT_NAME,
        radio_material=skin,
    )
    if HAND_OBJECT_NAME in scene.objects:
        scene.edit(remove=HAND_OBJECT_NAME, add=new_hand)
    else:
        scene.edit(add=new_hand)


# =============================
# RF SOLVE — per frame
# =============================

def process_frame(scene, solver, mesh_path, skin):
    swap_hand(scene, mesh_path, skin)

    paths = solver(
        scene=scene,
        max_depth=MAX_DEPTH,
        los=True,
        samples_per_src=50_000,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=False,
        diffraction=False,
    )

    result = paths.cfr(
        frequencies=subcarrier_freqs,
        num_time_steps=1,
        out_type="numpy",
    )

    H   = np.array(result).squeeze().flatten()[:NUM_SUBCARRIERS]
    CIR = np.fft.ifft(H)
    return H, CIR


# =============================
# SAVE / LOAD
# =============================

def save_rf_frame(H, CIR, idx):
    out_path = os.path.join(RF_OUTPUT_DIR, f"rf_frame_{idx:05d}.npz")
    np.savez(
        out_path,
        H_real   = np.real(H).astype(np.float32),
        H_imag   = np.imag(H).astype(np.float32),
        CIR_real = np.real(CIR).astype(np.float32),
        CIR_imag = np.imag(CIR).astype(np.float32),
    )


def load_rf(idx):
    path = os.path.join(RF_OUTPUT_DIR, f"rf_frame_{idx:05d}.npz")
    data = np.load(path)
    return (
        data["H_real"]   + 1j * data["H_imag"],
        data["CIR_real"] + 1j * data["CIR_imag"],
    )


# =============================
# VIDEO
# =============================

def generate_video(mesh_paths, start_idx, output_file, fps=10):
    total  = len(mesh_paths)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    canvas = FigureCanvasAgg(fig)
    writer = imageio.get_writer(output_file, fps=fps, macro_block_size=1)

    for i, mesh_path in enumerate(mesh_paths):
        idx = start_idx + i
        try:
            H, CIR = load_rf(idx)
        except Exception:
            continue

        ax1.clear(); ax2.clear()
        ax1.plot(np.abs(H));   ax1.set_title(f"CSI | Frame {idx}")
        ax2.plot(np.abs(CIR)); ax2.set_title(f"CIR | Frame {idx}")

        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3]
        writer.append_data(image)
        print(f"[video {i+1}/{total}] {os.path.basename(mesh_path)}")

    writer.close()
    plt.close(fig)


# =============================
# MAIN
# =============================

def main():
    # Output dirs
    for d in [RF_OUTPUT_DIR, PREFETCH_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    frames     = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))
    mesh_paths = [os.path.join(FRAMES_DIR, f) for f in frames]
    total      = len(mesh_paths)

    N     = 1000
    start = random.randint(0, total - N)
    end   = start + N
    subset_paths = mesh_paths[start:end]
    print(f"Using frames {start} → {end - 1}  ({N} frames)\n")

    # Build static scene once
    print("[init] Building static scene...")
    scene, solver, skin = build_scene()
    print("[init] Done.\n")

    # ── Start prefetch thread ──────────────────────────────────────────────
    # Queue size limits how far ahead the thread can get — keeps
    # PREFETCH_DIR from filling up with thousands of PLYs at once.
    frame_queue: queue.Queue = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)

    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(subset_paths, frame_queue),
        daemon=True,
    )
    prefetch_thread.start()

    # ── Main loop — consume from queue, run Sionna ─────────────────────────
    errors   = 0
    solved   = 0
    t_start  = time.time()

    while True:
        item = frame_queue.get()

        if item is _DONE:
            break

        queue_idx, prefetched_path = item

        # Absolute frame index into the dataset
        frame_idx = start + queue_idx

        if prefetched_path is None:
            errors += 1
            continue

        try:
            H, CIR = process_frame(scene, solver, prefetched_path, skin)
            save_rf_frame(H, CIR, frame_idx)
            solved += 1

            if solved % 10 == 0:
                elapsed  = time.time() - t_start
                fps_rate = solved / elapsed
                eta      = (N - solved) / fps_rate if fps_rate > 0 else 0
                print(f"[{solved}/{N}] frame {frame_idx} | "
                      f"{fps_rate:.2f} fr/s | ETA {eta:.0f}s")

        except Exception as e:
            print(f"[ERROR] Frame {frame_idx}: {e}")
            errors += 1

        finally:
            # Clean up the prefetched PLY immediately — no point keeping it
            try:
                os.remove(prefetched_path)
            except OSError:
                pass

    prefetch_thread.join()
    total_time = time.time() - t_start
    print(f"\nDone. {solved}/{N} frames in {total_time:.1f}s "
          f"({solved/total_time:.2f} fr/s) | {errors} errors")

    generate_video(
        subset_paths,
        start_idx=start,
        output_file="csi_cir_output_v1.mp4",
        fps=10,
    )

    # Clean up prefetch dir
    shutil.rmtree(PREFETCH_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()