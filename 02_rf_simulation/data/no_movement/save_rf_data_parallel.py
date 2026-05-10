import os
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

from multiprocessing import Pool, cpu_count
from sionna.rt import (
    load_scene_from_string, Transmitter, Receiver,
    PlanarArray, PathSolver, RadioMaterial
)
import shutil

# =============================
# CONFIG (UNCHANGED)
# =============================

FRAMES_DIR      = "/Users/dinesh/Documents/mtp/hand_models/no_movement/hand_frames_normalized"
RF_OUTPUT_DIR   = "/Users/dinesh/Documents/mtp/hand_models/no_movement/rf_output_parallel"

FREQUENCY       = 28e9
BANDWIDTH       = 400e6
NUM_SUBCARRIERS = 256
MAX_DEPTH       = 6

TX_POSITION = np.array([-0.106, 0.026, 0.014])
RX_POSITION = np.array([0.214, 0.026, 0.014])

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS
)).astype(np.float32)

os.makedirs(RF_OUTPUT_DIR, exist_ok=True)

# =============================
# XML SCENE (YOUR ORIGINAL — UNCHANGED)
# =============================

def generate_xml(mesh_path, idx):
    return f"""<scene version="2.1.0">
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
        <bsdf type="itu-radio-material" id="mat-wall-left-{idx}">
            <string name="type" value="concrete"/>
            <float name="thickness" value="0.2"/>
        </bsdf>
        <bsdf type="itu-radio-material" id="mat-wall-right-{idx}">
            <string name="type" value="concrete"/>
            <float name="thickness" value="0.2"/>
        </bsdf>

        <!-- Hand -->
        <shape type="ply">
            <string name="filename" value="{mesh_path}"/>
            <boolean name="face_normals" value="true"/>
            <ref id="mat-hand-{idx}" name="bsdf"/>
        </shape>

        <!-- Floor -->
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="5" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
            <ref id="mat-floor-{idx}" name="bsdf"/>
        </shape>

        <!-- Front wall -->
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="1" y="0" z="0" angle="90"/>
                <translate x="0" y="0.5" z="0"/>
            </transform>
            <ref id="mat-wall-front-{idx}" name="bsdf"/>
        </shape>

        <!-- Left wall -->
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="0" y="1" z="0" angle="90"/>
                <translate x="-2.0" y="0" z="0"/>
            </transform>
            <ref id="mat-wall-left-{idx}" name="bsdf"/>
        </shape>

        <!-- Right wall -->
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="0" y="1" z="0" angle="90"/>
                <translate x="2.0" y="0" z="0"/>
            </transform>
            <ref id="mat-wall-right-{idx}" name="bsdf"/>
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

    paths = solver(
        scene=scene,
        max_depth=MAX_DEPTH,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=True,
        diffraction=True
    )

    result = paths.cfr(
        frequencies=subcarrier_freqs,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy"
    )

    if isinstance(result, tuple):
        H_real = np.array(result[0]).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.array(result[1]).squeeze().flatten()[:NUM_SUBCARRIERS]
    else:
        H_real = np.real(result).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.imag(result).squeeze().flatten()[:NUM_SUBCARRIERS]

    H = H_real + 1j * H_imag
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
        frame_idx=np.int32(idx),
        freq_axis=subcarrier_freqs
    )

# =============================
# WORKER (PARALLEL)
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
    print(f"Total frames: {total}")

    num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} CPU cores")

    tasks = [(mesh_paths[i], i) for i in range(total)]

    with Pool(num_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, tasks)):
            if (i+1) % 10 == 0:
                print(f"[{i+1}/{total}] processed")

    print("All frames processed.")

# =============================
# RUN
# =============================

if __name__ == "__main__":
    main()