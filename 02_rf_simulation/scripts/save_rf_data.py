import os
import shutil
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib.pyplot as plt
from sionna.rt import (load_scene_from_string, Transmitter, Receiver,
                        PlanarArray, PathSolver, RadioMaterial)
from collections import deque
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
import shutil


# Configuration

FRAMES_DIR      = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_normalized"
FREQUENCY       = 28e9
BANDWIDTH       = 400e6
NUM_SUBCARRIERS = 256
MAX_DEPTH       = 6
WINDOW_SIZE     = 4
TX_POSITION = np.array([-0.149, -0.026, 0.031])
RX_POSITION = np.array([ 0.329, -0.026, 0.031])

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS
)).astype(np.float32)

# XML SCENE TEMPLATE — with floor and walls

def generate_xml(mesh_path, idx):
    # unique bsdf ids per frame to avoid Sionna registry collision
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

            <!-- Hand mesh -->
            <shape type="ply">
                <string name="filename" value="{mesh_path}"/>
                <boolean name="face_normals" value="true"/>
                <ref id="mat-hand-{idx}" name="bsdf"/>
            </shape>

            <!-- Floor: 5x5 flat plane at z=-0.5 -->
            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="5" z="1"/>
                    <translate x="0" y="0" z="-0.5"/>
                </transform>
                <ref id="mat-floor-{idx}" name="bsdf"/>
            </shape>

            <!-- Front wall: placed along +Y axis at y=0.5, height=1.0, width=5 -->
            <!-- Rectangle default normal is +Z; rotate 90 deg around X to make it vertical facing Y -->
            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="1" z="1"/>
                    <rotate x="1" y="0" z="0" angle="90"/>
                    <translate x="0" y="0.5" z="0"/>
                </transform>
                <ref id="mat-wall-front-{idx}" name="bsdf"/>
            </shape>

            <!-- Left side wall: placed at x=-2.0, height=1.0, depth=5 (larger distance) -->
            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="1" z="1"/>
                    <rotate x="0" y="1" z="0" angle="90"/>
                    <translate x="-2.0" y="0" z="0"/>
                </transform>
                <ref id="mat-wall-left-{idx}" name="bsdf"/>
            </shape>

            <!-- Right side wall: placed at x=+2.0, height=1.0, depth=5 (larger distance) -->
            <shape type="rectangle">
                <transform name="to_world">
                    <scale x="5" y="1" z="1"/>
                    <rotate x="0" y="1" z="0" angle="90"/>
                    <translate x="2.0" y="0" z="0"/>
                </transform>
                <ref id="mat-wall-right-{idx}" name="bsdf"/>
            </shape>
        </scene>"""


# Function to process each frame and extract CSI/CIR
def process_frame(mesh_path, idx):
    xml_str = generate_xml(mesh_path, idx)

    scene = load_scene_from_string(xml_str)
    scene.frequency = FREQUENCY

    if idx == 0:
        print("  Scene object keys:", list(scene.objects.keys()))

    # Fresh skin material with unique name every frame
    skin = RadioMaterial(
        name                  = f"skin_{idx}",
        relative_permittivity = 17.3,
        conductivity          = 25.6,
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
    scene.add(Receiver(name="rx",    position=RX_POSITION))

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
        frequencies      = subcarrier_freqs,
        num_time_steps   = 1,
        normalize_delays = True,
        normalize        = False,
        out_type         = "numpy"
    )

    # cfr returns (real, imag) tuple — not a single squeezable array
    if isinstance(result, tuple):
        H_real = np.array(result[0]).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.array(result[1]).squeeze().flatten()[:NUM_SUBCARRIERS]
    else:
        H_real = np.real(result).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.imag(result).squeeze().flatten()[:NUM_SUBCARRIERS]

    H   = H_real + 1j * H_imag
    CIR = np.fft.ifft(H)

    return H, CIR


RF_OUTPUT_DIR = "/Users/dinesh/Documents/mtp/hand_models/rf_output"

# =============================
# SAVE RF DATA PER FRAME
# =============================
def save_rf_frame(H, CIR, ply_frame_idx, output_dir=RF_OUTPUT_DIR):
    """
    Saves RF data for one frame as a .npz file.

    Naming convention matches the PLY files:
        PLY  : frame_000.ply  (0-based)
        NPZ  : rf_frame_000.npz

    Contents
    --------
    H           : (NUM_SUBCARRIERS,) complex  — CSI
    CIR         : (NUM_CIR_TAPS,)    complex  — channel impulse response
    H_real/imag : split real/imag for easy loading without complex dtype
    ply_frame_idx : int — which PLY frame this came from (0-based)
    json_frame    : int — corresponding JSON keypoint frame (= ply_frame_idx + 1)
    """
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"rf_frame_{ply_frame_idx:03d}.npz")

    np.savez(
        out_path,
        H_real        = np.real(H).astype(np.float32),
        H_imag        = np.imag(H).astype(np.float32),
        CIR_real      = np.real(CIR).astype(np.float32),
        CIR_imag      = np.imag(CIR).astype(np.float32),
        ply_frame_idx = np.int32(ply_frame_idx),
        json_frame    = np.int32(ply_frame_idx + 1),   # JSON is 1-based
        freq_axis     = subcarrier_freqs,
    )
    return out_path

# Plots for live visualization

plt.ion()
fig_live, (ax1_live, ax2_live) = plt.subplots(2, 1, figsize=(8, 6))

def update_plots(H, CIR, frame_name):
    ax1_live.clear()
    ax2_live.clear()

    ax1_live.plot(np.abs(H))
    ax1_live.set_title(f"CSI Magnitude - {frame_name}")
    ax1_live.set_xlabel("Subcarrier Index")
    ax1_live.set_ylabel("|H(f)|")

    ax2_live.plot(np.abs(CIR))
    ax2_live.set_title("CIR Magnitude")
    ax2_live.set_xlabel("Delay Bin")
    ax2_live.set_ylabel("|h(τ)|")

    plt.tight_layout()
    plt.pause(0.01)

# Function to generate video from all frames
def generate_video(mesh_paths, output_file="csi_cir_output.mp4", fps=10):
    print("Generating video...")

    total  = len(mesh_paths)
    fig_v, (ax1_v, ax2_v) = plt.subplots(2, 1, figsize=(8, 6))
    canvas = FigureCanvasAgg(fig_v)

    writer = imageio.get_writer(output_file, fps=fps, macro_block_size=1)

    for idx, mesh_path in enumerate(mesh_paths):
        frame_name = os.path.basename(mesh_path)
        H, CIR = process_frame(mesh_path, idx)
        save_rf_frame(H, CIR, ply_frame_idx=idx)

        ax1_v.clear()
        ax2_v.clear()

        ax1_v.plot(np.abs(H))
        ax1_v.set_title(
            f"CSI Magnitude  |  Frame {idx+1}/{total}  |  {frame_name}",
            fontsize=10
        )
        ax1_v.set_xlabel("Subcarrier Index")
        ax1_v.set_ylabel("|H(f)|")

        ax2_v.plot(np.abs(CIR))
        ax2_v.set_title(
            f"CIR Magnitude  |  Frame {idx+1}/{total}",
            fontsize=10
        )
        ax2_v.set_xlabel("Delay Bin")
        ax2_v.set_ylabel("|h(τ)|")

        # Progress bar at bottom of figure
        progress   = int(20 * (idx + 1) / total)
        fig_v.suptitle(
            f"mmWave 28 GHz — Hand + Floor  |  "
            f"Frame {idx+1}/{total}  "
            f"({'█' * progress}{'░' * (20 - progress)})",
            fontsize=9, y=0.02
        )

        fig_v.tight_layout(rect=[0, 0.04, 1, 1])

        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3]
        writer.append_data(image)

        print(f"  [{idx+1:3d}/{total}] {frame_name}")

    writer.close()
    plt.close(fig_v)
    print(f"Video saved as: {output_file}")

# main function to run the processing and visualization
def main():
    # clean output once
    if os.path.exists(RF_OUTPUT_DIR):
        shutil.rmtree(RF_OUTPUT_DIR)
    os.makedirs(RF_OUTPUT_DIR)


    frames = sorted(
        f for f in os.listdir(FRAMES_DIR)
        if f.endswith(".ply")
    )
    mesh_paths = [os.path.join(FRAMES_DIR, f) for f in frames]

    print("Starting dynamic CSI/CIR visualization...")

    # # dynamic sliding window processing
    # buffer = deque(maxlen=WINDOW_SIZE)
    # for idx, mesh_path in enumerate(mesh_paths):
    #     H, CIR = process_frame(mesh_path, idx)
    #     buffer.append((H, CIR))
    #     update_plots(H, CIR, os.path.basename(mesh_path))

    # sequential visualization of all frames
    for idx, mesh_path in enumerate(mesh_paths):
        H, CIR = process_frame(mesh_path, idx)
        save_rf_frame(H, CIR, ply_frame_idx=idx)
        update_plots(H, CIR, os.path.basename(mesh_path))
    print("Finished.")
    plt.ioff()
    plt.show()

    # video generation
    generate_video(mesh_paths, output_file="csi_cir_walls_floor_output.mp4", fps=10)


if __name__ == "__main__":
    main()