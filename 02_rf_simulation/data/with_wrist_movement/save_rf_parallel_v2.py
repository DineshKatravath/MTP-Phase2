import os
import numpy as np
import mitsuba as mi
import random
import multiprocessing
import shutil
import matplotlib.pyplot as plt
import imageio

from multiprocessing import Pool, cpu_count
from matplotlib.backends.backend_agg import FigureCanvasAgg

from sionna.rt import (
    load_scene_from_string, Transmitter, Receiver,
    PlanarArray, PathSolver, RadioMaterial
)

mi.set_variant("llvm_ad_mono_polarized")

# =============================
# CONFIG
# =============================

FRAMES_DIR = "/Users/dinesh/Documents/mtp/hand_models/with_wrist_movement/hand_frames_normalized"
RF_OUTPUT_DIR = "/Users/dinesh/Documents/mtp/hand_models/with_wrist_movement/rf_output_parallel_v2"

FREQUENCY = 28e9
BANDWIDTH = 400e6
NUM_SUBCARRIERS = 256
MAX_DEPTH = 6

TX_POSITION = np.array([-0.113, -0.006, -0.002])
RX_POSITION = np.array([0.208, -0.006, -0.002])

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS
)).astype(np.float32)

# =============================
# XML GENERATION
# =============================

def generate_xml(mesh_path, idx):
    return f"""<scene version="2.0.0">

            <!-- ===================== -->
            <!-- MATERIALS -->
            <!-- ===================== -->

            <bsdf type="itu-radio-material" id="mat-hand-{idx}">
                <string name="type" value="concrete"/>
                <float name="thickness" value="0.08"/>
            </bsdf>

            <bsdf type="itu-radio-material" id="mat-floor-{idx}">
                <string name="type" value="concrete"/>
                <float name="thickness" value="0.1"/>
            </bsdf>

            <bsdf type="itu-radio-material" id="mat-wall-front-{idx}">
                <string name="type" value="concrete"/>
                <float name="thickness" value="0.2"/>
            </bsdf>

            <bsdf type="itu-radio-material" id="mat-wall-back-{idx}">
                <string name="type" value="concrete"/>
                <float name="thickness" value="0.2"/>
            </bsdf>

            <bsdf type="itu-radio-material" id="mat-wall-left-{idx}">
                <string name="type" value="concrete"/>
                <float name="thickness" value="0.2"/>
            </bsdf>

            <bsdf type="itu-radio-material" id="mat-wall-right-{idx}">
                <string name="type" value="concrete"/>
                <float name="thickness" value="0.2"/>
            </bsdf>

            <bsdf type="itu-radio-material" id="mat-ceiling-{idx}">
                <string name="type" value="concrete"/>
                <float name="thickness" value="0.15"/>
            </bsdf>


            <!-- ===================== -->
            <!-- HAND MESH -->
            <!-- ===================== -->

            <shape type="ply">
                <string name="filename" value="{mesh_path}"/>
                <boolean name="face_normals" value="true"/>
                <ref id="mat-hand-{idx}" name="bsdf"/>
            </shape>


            <!-- ===================== -->
            <!-- FLOOR -->
            <!-- ===================== -->

            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="5" z="1"/>
                    <translate x="0" y="0" z="-0.15"/>
                </transform>
                <ref id="mat-floor-{idx}" name="bsdf"/>
            </shape>


            <!-- ===================== -->
            <!-- FRONT WALL (+Y) -->
            <!-- ===================== -->

            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="1" z="1"/>
                    <rotate x="1" y="0" z="0" angle="90"/>
                    <translate x="0" y="0.5" z="0"/>
                </transform>
                <ref id="mat-wall-front-{idx}" name="bsdf"/>
            </shape>


            <!-- ===================== -->
            <!-- BACK WALL (-Y) -->
            <!-- ===================== -->

            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="1" z="1"/>
                    <rotate x="1" y="0" z="0" angle="90"/>
                    <translate x="0" y="-0.5" z="0"/>
                </transform>
                <ref id="mat-wall-back-{idx}" name="bsdf"/>
            </shape>


            <!-- ===================== -->
            <!-- LEFT WALL (-X) -->
            <!-- ===================== -->

            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="1" z="1"/>
                    <rotate x="0" y="1" z="0" angle="90"/>
                    <translate x="-2.0" y="0" z="0"/>
                </transform>
                <ref id="mat-wall-left-{idx}" name="bsdf"/>
            </shape>


            <!-- ===================== -->
            <!-- RIGHT WALL (+X) -->
            <!-- ===================== -->

            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="1" z="1"/>
                    <rotate x="0" y="1" z="0" angle="90"/>
                    <translate x="2.0" y="0" z="0"/>
                </transform>
                <ref id="mat-wall-right-{idx}" name="bsdf"/>
            </shape>


            <!-- ===================== -->
            <!-- CEILING -->
            <!-- ===================== -->

            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="5" z="1"/>
                    <translate x="0" y="0" z="1.5"/>
                </transform>
                <ref id="mat-ceiling-{idx}" name="bsdf"/>
            </shape>

        </scene>"""

# =============================
# RF COMPUTATION
# =============================

def process_frame(mesh_path, idx):

    xml_str = generate_xml(mesh_path, idx)
    scene = load_scene_from_string(xml_str)
    scene.frequency = FREQUENCY

    skin = RadioMaterial(
        name=f"skin_{idx}",
        relative_permittivity=17.3,
        conductivity=25.6,
    )

    for obj in scene.objects.values():
        obj.radio_material = skin

    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )

    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )

    scene.add(Transmitter(name="tx", position=TX_POSITION))
    scene.add(Receiver(name="rx", position=RX_POSITION))

    solver = PathSolver()

    paths = solver(scene=scene, max_depth=MAX_DEPTH)

    result = paths.cfr(
        frequencies=subcarrier_freqs,
        num_time_steps=1,
        out_type="numpy"
    )

    H = np.array(result).squeeze().flatten()[:NUM_SUBCARRIERS]
    CIR = np.fft.ifft(H)

    return H, CIR

# =============================
# SAVE
# =============================

def save_rf_frame(H, CIR, idx):
    out_path = os.path.join(RF_OUTPUT_DIR, f"rf_frame_{idx:05d}.npz")

    np.savez(
        out_path,
        H_real=np.real(H).astype(np.float32),
        H_imag=np.imag(H).astype(np.float32),
        CIR_real=np.real(CIR).astype(np.float32),
        CIR_imag=np.imag(CIR).astype(np.float32),
    )

# =============================
# LOAD (for video)
# =============================

def load_rf(idx):
    path = os.path.join(RF_OUTPUT_DIR, f"rf_frame_{idx:05d}.npz")
    data = np.load(path)

    H = data["H_real"] + 1j * data["H_imag"]
    CIR = data["CIR_real"] + 1j * data["CIR_imag"]

    return H, CIR

# =============================
# WORKER
# =============================

def worker(args):
    mesh_path, idx = args
    try:
        H, CIR = process_frame(mesh_path, idx)
        save_rf_frame(H, CIR, idx)
        return idx
    except Exception as e:
        print(f"[ERROR] Frame {idx}: {e}")
        return None

# =============================
# VIDEO (NO RECOMPUTE)
# =============================

def generate_video(mesh_paths, start_idx, output_file, fps=10):

    total = len(mesh_paths)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    canvas = FigureCanvasAgg(fig)
    writer = imageio.get_writer(output_file, fps=fps, macro_block_size=1)

    for i, mesh_path in enumerate(mesh_paths):

        idx = start_idx + i
        frame_name = os.path.basename(mesh_path)

        try:
            H, CIR = load_rf(idx)
        except:
            continue

        ax1.clear()
        ax2.clear()

        ax1.plot(np.abs(H))
        ax1.set_title(f"CSI | Frame {idx}")

        ax2.plot(np.abs(CIR))
        ax2.set_title(f"CIR | Frame {idx}")

        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3]
        writer.append_data(image)

        print(f"[{i+1}/{total}] {frame_name}")

    writer.close()
    plt.close(fig)

# =============================
# MAIN
# =============================

def main():

    # clean output once
    if os.path.exists(RF_OUTPUT_DIR):
        shutil.rmtree(RF_OUTPUT_DIR)
    os.makedirs(RF_OUTPUT_DIR)

    frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))
    mesh_paths = [os.path.join(FRAMES_DIR, f) for f in frames]

    total = len(mesh_paths)

    # random contiguous 1000
    N = 1000
    start = random.randint(0, total - N)
    end = start + N

    subset_paths = mesh_paths[start:end]

    print(f"Using frames {start} → {end-1}")

    tasks = [(p, i) for i, p in enumerate(subset_paths, start)]

    # parallel compute
    with Pool(max(1, cpu_count() - 1)) as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, tasks)):
            if (i+1) % 10 == 0:
                print(f"[{i+1}/{len(tasks)}] processed")

    print("RF generation done")

    # video (no recompute)
    generate_video(
        subset_paths,
        start_idx=start,
        output_file="csi_cir_output.mp4",
        fps=10
    )

# =============================
# ENTRY
# =============================

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()