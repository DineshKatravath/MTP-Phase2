import os
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver

# Config

FRAMES_DIR = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_clean"
OUTPUT_FILE = "mmwave_gesture_sample.npy"
FREQUENCY = 28e9
BANDWIDTH = 100e6
NUM_SUBCARRIERS = 128
MAX_DEPTH = 6
TX_POSITION = [-0.5, 0, 0]
RX_POSITION = [0.5, 0, 0]

subcarrier_freqs = FREQUENCY + np.linspace(
    -BANDWIDTH/2,
    BANDWIDTH/2,
    NUM_SUBCARRIERS
).astype(np.float32)

frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))
all_csi = []
solver = PathSolver()

print("Processing frames...")
for i, frame_file in enumerate(frames):
    mesh_path = os.path.join(FRAMES_DIR, frame_file)

    xml = f"""<scene version="2.1.0">
        <bsdf type="itu-radio-material" id="mat-hand">
            <string name="type" value="glass"/>
            <float name="thickness" value="0.004"/>
        </bsdf>
        <shape type="ply" id="hand">
            <string name="filename" value="{mesh_path}"/>
            <boolean name="face_normals" value="true"/>
            <ref id="mat-hand" name="bsdf"/>
        </shape>
    </scene>"""

    xml_path = "temp_scene.xml"
    with open(xml_path, "w") as f:
        f.write(xml)

    scene = load_scene(xml_path)
    scene.frequency = FREQUENCY

    scene.objects["merged-shapes"].radio_material.relative_permittivity = 17.3
    scene.objects["merged-shapes"].radio_material.conductivity = 25.6

    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V"
    )

    tx = Transmitter(name="tx", position=TX_POSITION)
    rx = Receiver(name="rx", position=RX_POSITION)
    scene.add(tx)
    scene.add(rx)

    paths = solver(
        scene=scene,
        max_depth=MAX_DEPTH,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        diffraction=False
    )

    result = paths.cfr(
        frequencies=subcarrier_freqs,
        num_time_steps=1,
        normalize_delays=True,
        normalize=False,
        out_type="numpy"
    )

    # Debug on first frame
    if i == 0:
        print("result type:", type(result))
        if isinstance(result, tuple):
            print("result is tuple, len:", len(result))
            for j, r in enumerate(result):
                print(f"  result[{j}] type:", type(r), "shape:", getattr(r, 'shape', '?'))
        elif hasattr(result, 'shape'):
            print("result shape:", result.shape)
            print("result dtype:", result.dtype)

    # Handle all possible return formats
    if isinstance(result, tuple):
        if len(result) == 2:
            # (real, imag) tuple of arrays
            H_real = np.array(result[0]).squeeze()
            H_imag = np.array(result[1]).squeeze()
        else:
            raise ValueError(f"Unexpected tuple length from cfr(): {len(result)}")
    elif np.iscomplexobj(result):
        # Single complex numpy array
        H_real = np.real(result).squeeze()
        H_imag = np.imag(result).squeeze()
    else:
        # Real-only array (unexpected)
        print(f"WARNING: real-only result shape {result.shape}, assuming zero imaginary part")
        H_real = np.array(result).squeeze()
        H_imag = np.zeros_like(H_real)

    # Ensure shape is (NUM_SUBCARRIERS,)
    H_real = H_real.flatten()[:NUM_SUBCARRIERS]
    H_imag = H_imag.flatten()[:NUM_SUBCARRIERS]

    # Stack to (128, 2)
    H = np.stack([H_real, H_imag], axis=-1)
    all_csi.append(H)
    print(f"Processed: {frame_file}  H shape: {H.shape}")

# Formatting Dataset
dataset = np.array(all_csi)  # (num_frames, 128, 2)

np.save(OUTPUT_FILE, dataset)
print("\nDone.")
print("Final dataset shape:", dataset.shape)  # Expected: (num_frames, 128, 2)