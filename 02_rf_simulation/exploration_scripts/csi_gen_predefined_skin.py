import os
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sionna.rt import (load_scene, Transmitter, Receiver,
                        PlanarArray, PathSolver, RadioMaterial)

# =============================
# CONFIGURATION
# =============================
FRAMES_DIR      = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_clean"
OUTPUT_VIDEO    = "csi_cir_hand_28GHz.mp4"
OUTPUT_NPY      = "mmwave_gesture_sample.npy"
FREQUENCY       = 28e9
BANDWIDTH       = 100e6
NUM_SUBCARRIERS = 128
MAX_DEPTH       = 6
FPS             = 10
TX_POSITION     = [-0.5, 0, 0]
RX_POSITION     = [ 0.5, 0, 0]

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS
)).astype(np.float32)
freqs_ghz = subcarrier_freqs / 1e9
solver    = PathSolver()

# =============================
# XML — no id on shapes, unique bsdf ids per frame
# =============================
def make_xml(mesh_path, idx):
    # Use unique bsdf ids per frame to avoid Sionna's registry collision
    return f"""<scene version="2.1.0">
    <bsdf type="itu-radio-material" id="mat-hand-{idx}">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.004"/>
    </bsdf>
    <bsdf type="itu-radio-material" id="mat-floor-{idx}">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.1"/>
    </bsdf>
    <shape type="ply">
        <string name="filename" value="{mesh_path}"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-hand-{idx}" name="bsdf"/>
    </shape>
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="5" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
        <ref id="mat-floor-{idx}" name="bsdf"/>
    </shape>
</scene>"""

# =============================
# PROCESS ONE FRAME
# =============================
def process_frame(mesh_path, idx):
    xml      = make_xml(mesh_path, idx)
    xml_path = f"temp_scene_{idx}.xml"
    with open(xml_path, "w") as f:
        f.write(xml)

    scene = load_scene(xml_path)
    scene.frequency = FREQUENCY

    if idx == 0:
        print("  Scene object keys:", list(scene.objects.keys()))

    # Fresh material with unique name every frame
    skin = RadioMaterial(
        name                  = f"skin_{idx}",
        relative_permittivity = 17.3,
        conductivity          = 25.6,
    )
    for obj in scene.objects.values():
        obj.radio_material = skin

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                  vertical_spacing=0.5,
                                  horizontal_spacing=0.5,
                                  pattern="iso", polarization="V")
    scene.add(Transmitter(name="tx", position=TX_POSITION))
    scene.add(Receiver(name="rx",    position=RX_POSITION))

    paths = solver(scene=scene, max_depth=MAX_DEPTH,
                   los=True, specular_reflection=True,
                   diffuse_reflection=False, refraction=True,
                   diffraction=False)

    result = paths.cfr(frequencies=subcarrier_freqs,
                        num_time_steps=1, normalize_delays=True,
                        normalize=False, out_type="numpy")

    if isinstance(result, tuple):
        H_real = np.array(result[0]).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.array(result[1]).squeeze().flatten()[:NUM_SUBCARRIERS]
    else:
        H_real = np.real(result).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.imag(result).squeeze().flatten()[:NUM_SUBCARRIERS]

    H_complex = H_real + 1j * H_imag
    CIR       = np.fft.ifft(H_complex)
    delay_ns  = np.arange(NUM_SUBCARRIERS) / BANDWIDTH * 1e9

    os.remove(xml_path)
    return H_complex, CIR, delay_ns

# =============================
# RENDER FRAME TO IMAGE
# =============================
def render_frame_image(fig, axes, H, CIR, delay_ns,
                        frame_name, frame_idx, total):
    ax_csi, ax_cir, ax_phase, ax_hist = axes

    H_db    = 20 * np.log10(np.abs(H)  + 1e-12)
    H_ph    = np.angle(H, deg=True)
    CIR_mag = np.abs(CIR)

    ax_csi.clear()
    ax_csi.plot(freqs_ghz, H_db, color='#2196F3', linewidth=1.8)
    ax_csi.fill_between(freqs_ghz, H_db.min(), H_db,
                         alpha=0.15, color='#2196F3')
    ax_csi.set_title("CSI — Magnitude (dB)", color='white', fontsize=11)
    ax_csi.set_xlabel("Frequency (GHz)", color='#aaaaaa', fontsize=9)
    ax_csi.set_ylabel("Magnitude (dB)", color='#aaaaaa', fontsize=9)
    ax_csi.set_xlim(freqs_ghz[0], freqs_ghz[-1])
    ax_csi.tick_params(colors='#aaaaaa')
    ax_csi.grid(True, color='#2a2a2a', linewidth=0.6)

    ax_cir.clear()
    ml, sl, bl = ax_cir.stem(delay_ns, CIR_mag,
                              linefmt='#00BCD4',
                              markerfmt='C0o',
                              basefmt='#444444')
    plt.setp(sl, linewidth=1.2)
    plt.setp(ml, markersize=4)
    ax_cir.set_title("CIR — Channel Impulse Response", color='white', fontsize=11)
    ax_cir.set_xlabel("Delay (ns)", color='#aaaaaa', fontsize=9)
    ax_cir.set_ylabel("|h(τ)|", color='#aaaaaa', fontsize=9)
    ax_cir.tick_params(colors='#aaaaaa')
    ax_cir.grid(True, color='#2a2a2a', linewidth=0.6)

    ax_phase.clear()
    ax_phase.plot(freqs_ghz, H_ph, color='#CE93D8', linewidth=1.5)
    ax_phase.set_title("CSI — Phase", color='white', fontsize=11)
    ax_phase.set_xlabel("Frequency (GHz)", color='#aaaaaa', fontsize=9)
    ax_phase.set_ylabel("Phase (°)", color='#aaaaaa', fontsize=9)
    ax_phase.set_xlim(freqs_ghz[0], freqs_ghz[-1])
    ax_phase.set_ylim(-185, 185)
    ax_phase.tick_params(colors='#aaaaaa')
    ax_phase.grid(True, color='#2a2a2a', linewidth=0.6)

    ax_hist.clear()
    ax_hist.bar(delay_ns, CIR_mag,
                width=(delay_ns[1]-delay_ns[0])*0.8,
                color='#FF9800', alpha=0.8, edgecolor='none')
    ax_hist.set_title("CIR — Multipath Power Profile", color='white', fontsize=11)
    ax_hist.set_xlabel("Delay (ns)", color='#aaaaaa', fontsize=9)
    ax_hist.set_ylabel("Power",      color='#aaaaaa', fontsize=9)
    ax_hist.tick_params(colors='#aaaaaa')
    ax_hist.grid(True, color='#2a2a2a', linewidth=0.6, axis='y')

    for ax in axes:
        ax.set_facecolor('#161b22')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333333')

    fig.suptitle(
        f"mmWave 28 GHz — Hand Gesture CSI/CIR  |  "
        f"Frame {frame_idx+1}/{total}  |  {frame_name}\n"
        f"Skin: εr=17.3, σ=25.6 S/m  (ITU-R P.2040)",
        color='white', fontsize=11, fontweight='bold', y=0.99
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

# =============================
# MAIN
# =============================
def main():
    frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))
    total  = len(frames)
    print(f"Found {total} frames")
    print(f"Material: Human skin 28 GHz  εr=17.3  σ=25.6 S/m\n")

    all_csi = []
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor='#0d1117')
    axes   = axes.flatten()
    canvas = FigureCanvasAgg(fig)

    writer = imageio.get_writer(OUTPUT_VIDEO, fps=FPS, macro_block_size=1)
    print(f"Generating video: {OUTPUT_VIDEO}\n")

    for idx, frame_file in enumerate(frames):
        mesh_path = os.path.join(FRAMES_DIR, frame_file)
        H, CIR, delay_ns = process_frame(mesh_path, idx)
        all_csi.append(np.stack([H.real, H.imag], axis=-1))

        render_frame_image(fig, axes, H, CIR, delay_ns,
                           frame_file, idx, total)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3]
        writer.append_data(image)

        print(f"  [{idx+1:3d}/{total}] {frame_file} "
              f"| peak CIR: {np.abs(CIR).max():.4f} "
              f"| mean |H|: {np.abs(H).mean():.4f}")

    writer.close()
    print(f"\nVideo saved : {OUTPUT_VIDEO}")

    dataset = np.array(all_csi)
    np.save(OUTPUT_NPY, dataset)
    print(f"Dataset saved: {OUTPUT_NPY}  shape: {dataset.shape}")

    # ── Summary plot ──────────────────────────────────────────────────────────
    dataset_c  = dataset[:,:,0] + 1j*dataset[:,:,1]
    mean_mag   = np.mean(20*np.log10(np.abs(dataset_c)+1e-12), axis=0)
    std_mag    = np.std( 20*np.log10(np.abs(dataset_c)+1e-12), axis=0)
    mean_cir   = np.mean(np.abs(np.fft.ifft(dataset_c, axis=1)), axis=0)
    heatmap    = 20*np.log10(np.abs(dataset_c)+1e-12)
    cmap_lines = plt.cm.plasma(np.linspace(0, 1, min(20, total)))
    delay_ns   = np.arange(NUM_SUBCARRIERS) / BANDWIDTH * 1e9

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8), facecolor='#0d1117')
    fig2.suptitle(
        f"CSI/CIR Summary — {total} frames | 28 GHz | Skin (εr=17.3, σ=25.6)",
        color='white', fontsize=13, fontweight='bold'
    )
    axes2 = axes2.flatten()

    axes2[0].plot(freqs_ghz, mean_mag, color='#2196F3', linewidth=2, label='Mean')
    axes2[0].fill_between(freqs_ghz, mean_mag-std_mag, mean_mag+std_mag,
                           alpha=0.3, color='#2196F3', label='±1 std')
    axes2[0].set_title("Mean CFR Magnitude", color='white')
    axes2[0].set_xlabel("Frequency (GHz)", color='#aaaaaa')
    axes2[0].set_ylabel("dB", color='#aaaaaa')
    axes2[0].legend(facecolor='#1c2128', edgecolor='#444', labelcolor='white')

    for j in range(min(20, total)):
        axes2[1].plot(freqs_ghz,
                      20*np.log10(np.abs(dataset_c[j])+1e-12),
                      color=cmap_lines[j], linewidth=0.9, alpha=0.7)
    axes2[1].set_title("CFR Magnitude — first 20 frames", color='white')
    axes2[1].set_xlabel("Frequency (GHz)", color='#aaaaaa')
    axes2[1].set_ylabel("dB", color='#aaaaaa')

    axes2[2].stem(delay_ns, mean_cir,
                  linefmt='#00BCD4', markerfmt='C0o', basefmt='#444')
    axes2[2].set_title("Mean CIR across all frames", color='white')
    axes2[2].set_xlabel("Delay (ns)", color='#aaaaaa')
    axes2[2].set_ylabel("|h(τ)|", color='#aaaaaa')

    im = axes2[3].imshow(heatmap, aspect='auto', cmap='plasma',
                          extent=[freqs_ghz[0], freqs_ghz[-1], total, 0])
    axes2[3].set_title("CSI Magnitude Heatmap (all frames)", color='white')
    axes2[3].set_xlabel("Frequency (GHz)", color='#aaaaaa')
    axes2[3].set_ylabel("Frame index", color='#aaaaaa')
    fig2.colorbar(im, ax=axes2[3], label='dB')

    for ax in axes2:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#aaaaaa')
        ax.grid(True, color='#2a2a2a', linewidth=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333333')

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig("csi_cir_summary.png", dpi=150, bbox_inches='tight',
                 facecolor=fig2.get_facecolor())
    plt.show()
    print("Summary saved: csi_cir_summary.png")

if __name__ == "__main__":
    main()