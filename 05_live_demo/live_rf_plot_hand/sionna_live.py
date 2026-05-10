"""
sionna_live.py
==============
Watches hand_frames_live/ for new PLY files, runs the full pipeline:
  clean → normalize → Sionna RF → save .npz

No matplotlib here — plotting is handled by rf_plotter.py running
in the main process so macOS GUI works correctly.
"""

import os
import time
import shutil
import numpy as np
import trimesh

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

from sionna.rt import (load_scene_from_string, Transmitter, Receiver,
                       PlanarArray, PathSolver, RadioMaterial)

# ------------------------------------------------------------------ CONFIG ---

BASE            = "/Users/dinesh/Documents/mtp/hand_models/live_rf_plot_hand"
LIVE_DIR        = os.path.join(BASE, "hand_frames_live")
CLEAN_DIR       = os.path.join(BASE, "hand_frames_live_clean")
NORM_DIR        = os.path.join(BASE, "hand_frames_live_normalized")
RF_DIR          = os.path.join(BASE, "rf_output")
PROCESSED_DIR   = os.path.join(LIVE_DIR, "processed")

FREQUENCY       = 28e9
BANDWIDTH       = 400e6
NUM_SUBCARRIERS = 64
# the number of subcarriers determines the resolution of the channel impulse response (CIR) in the time domain. With 64 subcarriers, we can capture multipath components that are spaced at least 1/(400e6) = 2.5 ns apart, which corresponds to a path length difference of about 0.75 meters. 
# This should be sufficient for capturing the main multipath components in a hand gesture scenario, while keeping the computational load manageable for real-time processing.

MAX_DEPTH       = 6
# later we can keep it as 6, as this represents the maximum number of interactions (reflections, diffractions, etc.) that a ray can undergo before reaching the receiver. 
# Setting it to 2 is a simplification for faster processing during development, but increasing it to 6 would allow for more complex and realistic scenarios to be simulated, albeit with increased computational time.
TARGET_SPAN_M   = 0.20
CLEARANCE_RATIO = 0.75
IDLE_TIMEOUT    = 30.0   # sionna ray-tracing takes several seconds per frame
POLL_INTERVAL   = 0.05

os.makedirs(LIVE_DIR, exist_ok=True)

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH / 2, BANDWIDTH / 2, NUM_SUBCARRIERS
)).astype(np.float32)

TX_POSITION  = None
RX_POSITION  = None
scale_factor = None

# ------------------------------------------------------------------- CLEAN ---

def clean_frame(src, dst):
    mesh = trimesh.load(src, process=False)
    mesh.export(dst)

# ---------------------------------------------------------------- NORMALISE ---

def compute_scale_factor(clean_path):
    mesh  = trimesh.load(clean_path, process=False)
    verts = np.array(mesh.vertices)
    span  = verts.max() - verts.min()
    sf    = TARGET_SPAN_M / span
    print(f"[calib] span={span:.4f} units  scale={sf:.6f} m/unit")
    return sf

def normalize_frame(src, dst, sf):
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
    print(f"[calib] separation={rx[0]-tx[0]:.4f}m  hand fills ~{span/(rx[0]-tx[0])*100:.0f}%")
    return tx, rx

# ------------------------------------------------------------------ SIONNA ---

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
        <shape type="ply">
            <string name="filename" value="{mesh_path}"/>
            <boolean name="face_normals" value="true"/>
            <ref id="mat-hand-{idx}" name="bsdf"/>
        </shape>
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="5" z="1"/>
                <translate x="0" y="0" z="-0.15"/>
            </transform>
            <ref id="mat-floor-{idx}" name="bsdf"/>
        </shape>
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="1" y="0" z="0" angle="90"/>
                <translate x="0" y="0.5" z="0"/>
            </transform>
            <ref id="mat-wall-front-{idx}" name="bsdf"/>
        </shape>
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="0" y="1" z="0" angle="90"/>
                <translate x="-2.0" y="0" z="0"/>
            </transform>
            <ref id="mat-wall-left-{idx}" name="bsdf"/>
        </shape>
        <shape type="rectangle">
            <transform name="to_world">
                <scale x="5" y="1" z="1"/>
                <rotate x="0" y="1" z="0" angle="90"/>
                <translate x="2.0" y="0" z="0"/>
            </transform>
            <ref id="mat-wall-right-{idx}" name="bsdf"/>
        </shape>
    </scene>"""


def process_rf(norm_path, idx, tx, rx):
    scene = load_scene_from_string(generate_xml(norm_path, idx))
    scene.frequency = FREQUENCY

    skin = RadioMaterial(
        name=f"skin_{idx}",
        relative_permittivity=17.3,
        conductivity=25.6,
    )
    for obj in scene.objects.values():
        obj.radio_material = skin

    arr = PlanarArray(num_rows=1, num_cols=1,
                      vertical_spacing=0.5, horizontal_spacing=0.5,
                      pattern="iso", polarization="V")
    scene.tx_array = arr
    scene.rx_array = arr

    scene.add(Transmitter(name="tx", position=tx))
    scene.add(Receiver(name="rx",    position=rx))

    paths = PathSolver()(
        scene=scene, max_depth=MAX_DEPTH,
        los=True, specular_reflection=True,
        diffuse_reflection=True, refraction=True, diffraction=True,
    )
    # we can enable more interactions (reflections, diffractions, etc.) by setting the corresponding flags to True,
    #  but for faster processing during development, we keep them disabled.

    result = paths.cfr(
        frequencies=subcarrier_freqs, num_time_steps=1,
        normalize_delays=True, normalize=False, out_type="numpy",
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

# ----------------------------------------------------------------- SAVE RF ---

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

# -------------------------------------------------------------------- MAIN ---

def main():
    global TX_POSITION, RX_POSITION, scale_factor

    print("=" * 60)
    print("sionna_live.py started")
    print(f"Watching: {LIVE_DIR}")
    print("=" * 60)

    # DO NOT delete LIVE_DIR (important)
    for d in [CLEAN_DIR, NORM_DIR, RF_DIR, PROCESSED_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    os.makedirs(LIVE_DIR, exist_ok=True)

    calibrated = False
    last_seen_t = None
    idx = 0

    try:
        while True:

            files = sorted([f for f in os.listdir(LIVE_DIR) if f.endswith(".ply")])

            if not files:
                if last_seen_t and (time.time() - last_seen_t > IDLE_TIMEOUT):
                    print("\nNo new frames — exiting.")
                    break

                time.sleep(POLL_INTERVAL)
                continue

            # fname = files[0]# for proceesing in  order

            #  for latest frame processing (skip backlog, if any)
            fname = files[-1]
            candidate = os.path.join(LIVE_DIR, fname)

            # delete older frames (avoiding backlog)
            for f in files[:-1]:
                try:
                    os.remove(os.path.join(LIVE_DIR, f))
                except:
                    pass

            print(f"[DEBUG] Processing: {fname}")

            clean_path = os.path.join(CLEAN_DIR, fname)
            norm_path  = os.path.join(NORM_DIR, fname)

            # --- CLEAN ---
            try:
                clean_frame(candidate, clean_path)
            except Exception as e:
                print(f"[clean error] {fname}: {e}")
                os.rename(candidate, candidate + ".err")
                continue

            # --- CALIBRATE (once) ---
            if not calibrated:
                try:
                    scale_factor = compute_scale_factor(clean_path)
                    normalize_frame(clean_path, norm_path, scale_factor)
                    TX_POSITION, RX_POSITION = calibrate_tx_rx(norm_path)
                    calibrated = True
                except Exception as e:
                    print(f"[calibration error]: {e}")
                    os.rename(candidate, candidate + ".err")
                    continue

            # --- NORMALIZE ---
            try:
                center, span = normalize_frame(clean_path, norm_path, scale_factor)
            except Exception as e:
                print(f"[norm error] {fname}: {e}")
                os.rename(candidate, candidate + ".err")
                continue

            # --- RF ---
            try:
                print(f"[{idx:04d}] centre={center.round(3)} span={span:.4f}", end="  ")
                start = time.time()
                H, CIR = process_rf(norm_path, idx, TX_POSITION, RX_POSITION)
                print(f"RF done ({time.time()-start:.2f}s)", end="  ")
            except Exception as e:
                print(f"\n[sionna error] {fname}: {e}")
                os.rename(candidate, candidate + ".err")
                continue

            # --- SAVE ---
            rf_path = save_rf(H, CIR, idx)
            print(f"→ {os.path.basename(rf_path)}")

            # --- MOVE PROCESSED ---
            shutil.move(candidate, os.path.join(PROCESSED_DIR, fname))

            last_seen_t = time.time()
            idx += 1

    except KeyboardInterrupt:
        print("\nStopped manually.")

    print(f"\nDone. Processed {idx} frames.")

if __name__ == "__main__":
    main()