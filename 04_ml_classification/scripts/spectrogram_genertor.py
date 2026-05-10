import os
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sionna.rt import (load_scene_from_string, Transmitter, Receiver,
                        PlanarArray, PathSolver, RadioMaterial)
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg

# =============================
# CONFIGURATION
# =============================
FRAMES_DIR      = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_normalized"
FREQUENCY       = 28e9
BANDWIDTH       = 400e6          # Hz — sets CIR delay resolution
NUM_SUBCARRIERS = 256
MAX_DEPTH       = 6

TX_POSITION = np.array([-0.143, 0.020, -0.020])
RX_POSITION = np.array([0.167, 0.020, -0.020])

# Δτ_delay = 1/BW — fixed by RF bandwidth only, not sampling rate
DELTA_TAU_DELAY = 1.0 / BANDWIDTH       # seconds per CIR tap (2.5 ns @ 400 MHz)
NUM_CIR_TAPS    = NUM_SUBCARRIERS        # IFFT length = # subcarriers

subcarrier_freqs = (FREQUENCY + np.linspace(
    -BANDWIDTH / 2, BANDWIDTH / 2, NUM_SUBCARRIERS
)).astype(np.float32)

# Slow-time spacing = 1/FPS — provided by user at runtime


# =============================
# XML SCENE TEMPLATE
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

    <!-- Front wall at y=0.5 -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="1" y="0" z="0" angle="90"/>
            <translate x="0" y="0.5" z="0"/>
        </transform>
        <ref id="mat-wall-front-{idx}" name="bsdf"/>
    </shape>

    <!-- Left side wall at x=-2.0 -->
    <shape type="rectangle">
        <transform name="to_world">
            <scale x="5" y="1" z="1"/>
            <rotate x="0" y="1" z="0" angle="90"/>
            <translate x="-2.0" y="0" z="0"/>
        </transform>
        <ref id="mat-wall-left-{idx}" name="bsdf"/>
    </shape>

    <!-- Right side wall at x=+2.0 -->
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
# FRAME PROCESSING
# =============================
def process_frame(mesh_path, idx):
    xml_str = generate_xml(mesh_path, idx)
    scene   = load_scene_from_string(xml_str)
    scene.frequency = FREQUENCY

    skin = RadioMaterial(
        name                  = f"skin_{idx}",
        relative_permittivity = 17.3,
        conductivity          = 25.6,
    )
    for obj in scene.objects.values():
        obj.radio_material = skin

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

    scene.add(Transmitter(name="tx", position=TX_POSITION))
    scene.add(Receiver(name="rx",    position=RX_POSITION))

    paths = PathSolver()(
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

    if isinstance(result, tuple):
        H_real = np.array(result[0]).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.array(result[1]).squeeze().flatten()[:NUM_SUBCARRIERS]
    else:
        H_real = np.real(result).squeeze().flatten()[:NUM_SUBCARRIERS]
        H_imag = np.imag(result).squeeze().flatten()[:NUM_SUBCARRIERS]

    H   = H_real + 1j * H_imag      # (NUM_SUBCARRIERS,)
    CIR = np.fft.ifft(H)            # (NUM_CIR_TAPS,)
    return H, CIR


# =============================
# BUILD SLOW-TIME AXIS
# =============================
def build_slow_time_axis(frame_indices, blender_fps):
    # t[n] = frame_index / FPS  (seconds)
    return np.array(frame_indices, dtype=float) / blender_fps


# =============================
# COLLECT FRAMES INTO MATRICES
# =============================
def collect_frames(mesh_paths, frame_indices, blender_fps):
    """
    Returns
    -------
    H_mat          : (N, NUM_SUBCARRIERS)  — rows = frames (slow time), cols = subcarriers
    CIR_mat        : (N, NUM_CIR_TAPS)    — rows = frames (slow time), cols = delay taps
    slow_time_axis : (N,) seconds
    """
    N              = len(frame_indices)
    H_mat          = np.zeros((N, NUM_SUBCARRIERS), dtype=complex)
    CIR_mat        = np.zeros((N, NUM_CIR_TAPS),    dtype=complex)
    slow_time_axis = build_slow_time_axis(frame_indices, blender_fps)

    print(f"\n  Blender FPS = {blender_fps}")
    print(f"  Δt_slow     = 1/{blender_fps:.0f} s = {1e6/blender_fps:.2f} µs / frame")
    print(f"  Δτ_delay    = 1/BW = {DELTA_TAU_DELAY*1e9:.2f} ns / tap\n")

    for row, (fi, mesh_path) in enumerate(zip(frame_indices, mesh_paths)):
        t_us = slow_time_axis[row] * 1e6
        print(f"  [{row+1:3d}/{N}]  frame {fi:4d}  "
              f"t = {t_us:10.2f} µs  →  {os.path.basename(mesh_path)}")
        H, CIR              = process_frame(mesh_path, fi)
        H_mat[row, :]       = H       # row = one slow-time snapshot
        CIR_mat[row, :]     = CIR

    return H_mat, CIR_mat, slow_time_axis


# =============================
# SHARED AXIS HELPERS
# =============================
def _make_frame_yticks(ax_left, ax_right, frame_indices, slow_us, N):
    """
    Left  Y : frame index
    Right Y : slow time in µs
    """
    tick_step = max(1, N // 20)
    tick_rows = np.arange(0, N, tick_step)

    ax_left.set_yticks(tick_rows)
    ax_left.set_yticklabels([f"{frame_indices[r]}" for r in tick_rows], fontsize=7)
    ax_left.set_ylabel("Frame Index  (slow time ↑)", fontsize=9)

    ax_right.set_ylim(ax_left.get_ylim())
    ax_right.set_yticks(tick_rows)
    ax_right.set_yticklabels([f"{slow_us[r]:.0f}" for r in tick_rows], fontsize=7)
    ax_right.set_ylabel("Slow Time (µs)", fontsize=9)


def _make_subcarrier_xticks(ax):
    """X-axis for CSI: subcarrier index + freq offset in MHz on top."""
    freq_MHz = np.linspace(-BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS) / 1e6
    tick_step = max(1, NUM_SUBCARRIERS // 8)
    tick_cols = np.arange(0, NUM_SUBCARRIERS, tick_step)

    ax.set_xticks(tick_cols)
    ax.set_xticklabels([f"{int(c)}" for c in tick_cols], fontsize=7)
    ax.set_xlabel("Subcarrier Index  (amplitude →)", fontsize=9)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(tick_cols)
    ax_top.set_xticklabels([f"{freq_MHz[c]:.0f}" for c in tick_cols], fontsize=7)
    ax_top.set_xlabel("Freq Offset (MHz)", fontsize=9)


def _make_tap_xticks(ax):
    """X-axis for CIR: delay tap index + delay in ns on top."""
    delay_ns = np.arange(NUM_CIR_TAPS) * DELTA_TAU_DELAY * 1e9
    tick_step = max(1, NUM_CIR_TAPS // 8)
    tick_cols = np.arange(0, NUM_CIR_TAPS, tick_step)

    ax.set_xticks(tick_cols)
    ax.set_xticklabels([f"{int(c)}" for c in tick_cols], fontsize=7)
    ax.set_xlabel("Delay Tap Index  (amplitude →)", fontsize=9)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(tick_cols)
    ax_top.set_xticklabels([f"{delay_ns[c]:.1f}" for c in tick_cols], fontsize=7)
    ax_top.set_xlabel("Delay τ (ns)", fontsize=9)


# =============================
# PLOT SPECTROGRAMS
# =============================
def plot_spectrograms(H_mat, CIR_mat, slow_time_axis, blender_fps,
                      frame_indices, title_prefix="", save_path=None, show=True):
    """
    H_mat   : (N, NUM_SUBCARRIERS)  — rows = frames, cols = subcarriers
    CIR_mat : (N, NUM_CIR_TAPS)    — rows = frames, cols = delay taps

    Layout
    ------
    X-axis : amplitude axis
               CSI → subcarrier index  (top: freq offset MHz)
               CIR → delay tap index   (top: delay in ns)
    Y-axis : slow time
               Left  → frame index
               Right → time in µs
    Color  : |H| or |h| in dB
    """
    N       = H_mat.shape[0]
    slow_us = slow_time_axis * 1e6

    csi_dB = 20 * np.log10(np.abs(H_mat)   + 1e-12)   # (N, NUM_SUBCARRIERS)
    cir_dB = 20 * np.log10(np.abs(CIR_mat) + 1e-12)   # (N, NUM_CIR_TAPS)

    # imshow extent: [x_left, x_right, y_bottom, y_top]
    # X in tap/subcarrier index units, Y in row index units
    csi_extent = [-0.5, NUM_SUBCARRIERS - 0.5, -0.5, N - 0.5]
    cir_extent = [-0.5, NUM_CIR_TAPS    - 0.5, -0.5, N - 0.5]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, width_ratios=[20, 1], hspace=0.65, wspace=0.08)

    ax_csi      = fig.add_subplot(gs[0, 0])
    ax_cbar_csi = fig.add_subplot(gs[0, 1])
    ax_cir      = fig.add_subplot(gs[1, 0])
    ax_cbar_cir = fig.add_subplot(gs[1, 1])

    # ── CSI ───────────────────────────────────────────────────────────────────
    im_csi = ax_csi.imshow(
        csi_dB,
        aspect="auto", origin="lower", cmap="viridis",
        extent=csi_extent
    )
    ax_csi.set_title(f"{title_prefix}CSI Spectrogram  —  color = |H(f, t)| [dB]",
                     fontsize=10)
    _make_subcarrier_xticks(ax_csi)
    ax_csi_right = ax_csi.twinx()
    _make_frame_yticks(ax_csi, ax_csi_right, frame_indices, slow_us, N)
    plt.colorbar(im_csi, cax=ax_cbar_csi, label="|H| dB")

    # ── CIR ───────────────────────────────────────────────────────────────────
    im_cir = ax_cir.imshow(
        cir_dB,
        aspect="auto", origin="lower", cmap="plasma",
        extent=cir_extent
    )
    ax_cir.set_title(f"{title_prefix}CIR Spectrogram  —  color = |h(τ, t)| [dB]",
                     fontsize=10)
    _make_tap_xticks(ax_cir)
    ax_cir_right = ax_cir.twinx()
    _make_frame_yticks(ax_cir, ax_cir_right, frame_indices, slow_us, N)
    plt.colorbar(im_cir, cax=ax_cbar_cir, label="|h| dB")

    fig.suptitle(
        f"mmWave 28 GHz  |  BW = {BANDWIDTH/1e6:.0f} MHz  |  "
        f"Blender FPS = {blender_fps}  →  Δt_slow = {1e6/blender_fps:.2f} µs/frame  |  "
        f"Δτ_delay = {DELTA_TAU_DELAY*1e9:.2f} ns/tap  |  N = {N} frames",
        fontsize=9
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Spectrogram saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# =============================
# ROLLING-WINDOW VIDEO
# =============================
def generate_spectrogram_video(all_H, all_CIR, all_frame_indices, blender_fps,
                                window_size, output_file, fps=5):
    """
    Rolls a window of `window_size` frames and writes a spectrogram video.

    all_H   : list of length total_frames, each entry shape (NUM_SUBCARRIERS,)
    all_CIR : list of length total_frames, each entry shape (NUM_CIR_TAPS,)
    """
    print(f"\nGenerating rolling spectrogram video  "
          f"(window = {window_size} frames, video fps = {fps}) ...")

    N        = len(all_H)
    dt_s     = 1.0 / blender_fps
    delay_ns = np.arange(NUM_CIR_TAPS) * DELTA_TAU_DELAY * 1e9
    freq_MHz = np.linspace(-BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS) / 1e6

    fig    = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    writer = imageio.get_writer(output_file, fps=fps, macro_block_size=1)

    for start in range(N - window_size + 1):
        end     = start + window_size
        indices = list(range(start, end))        # column positions in window

        # Stack rows = frames, cols = taps/subcarriers
        H_win   = np.stack([all_H[i]   for i in indices], axis=0)   # (W, NUM_SUB)
        CIR_win = np.stack([all_CIR[i] for i in indices], axis=0)   # (W, NUM_TAPS)

        slow_us   = np.array(indices) * dt_s * 1e6
        W         = len(indices)
        tick_step = max(1, W // 10)
        tick_rows = np.arange(0, W, tick_step)

        csi_extent = [-0.5, NUM_SUBCARRIERS - 0.5, -0.5, W - 0.5]
        cir_extent = [-0.5, NUM_CIR_TAPS    - 0.5, -0.5, W - 0.5]

        fig.clf()
        gs  = gridspec.GridSpec(2, 2, width_ratios=[20, 1], hspace=0.65, wspace=0.08)
        ax1 = fig.add_subplot(gs[0, 0]);  cb1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0]);  cb2 = fig.add_subplot(gs[1, 1])

        # ── CSI window ────────────────────────────────────────────────────────
        im1 = ax1.imshow(
            20*np.log10(np.abs(H_win)+1e-12),
            aspect="auto", origin="lower", cmap="viridis",
            extent=csi_extent
        )
        ax1.set_title(f"CSI Spectrogram  |  frames {all_frame_indices[start]}–"
                      f"{all_frame_indices[end-1]}", fontsize=9)

        # X: subcarrier index (bottom) + freq MHz (top)
        sub_tick_step = max(1, NUM_SUBCARRIERS // 8)
        sub_ticks = np.arange(0, NUM_SUBCARRIERS, sub_tick_step)
        ax1.set_xticks(sub_ticks)
        ax1.set_xticklabels([f"{int(c)}" for c in sub_ticks], fontsize=6)
        ax1.set_xlabel("Subcarrier Index  (amplitude →)", fontsize=8)
        ax1_top = ax1.twiny()
        ax1_top.set_xlim(ax1.get_xlim())
        ax1_top.set_xticks(sub_ticks)
        ax1_top.set_xticklabels([f"{freq_MHz[c]:.0f}" for c in sub_ticks], fontsize=6)
        ax1_top.set_xlabel("Freq Offset (MHz)", fontsize=8)

        # Y: frame index (left) + slow time µs (right)
        ax1.set_yticks(tick_rows)
        ax1.set_yticklabels(
            [f"{all_frame_indices[start + r]}" for r in tick_rows], fontsize=6)
        ax1.set_ylabel("Frame Index  (slow time ↑)", fontsize=8)
        ax1r = ax1.twinx()
        ax1r.set_ylim(ax1.get_ylim())
        ax1r.set_yticks(tick_rows)
        ax1r.set_yticklabels([f"{slow_us[r]:.0f}" for r in tick_rows], fontsize=6)
        ax1r.set_ylabel("Slow Time (µs)", fontsize=8)

        plt.colorbar(im1, cax=cb1, label="|H| dB")

        # ── CIR window ────────────────────────────────────────────────────────
        im2 = ax2.imshow(
            20*np.log10(np.abs(CIR_win)+1e-12),
            aspect="auto", origin="lower", cmap="plasma",
            extent=cir_extent
        )
        ax2.set_title(f"CIR Spectrogram  |  frames {all_frame_indices[start]}–"
                      f"{all_frame_indices[end-1]}", fontsize=9)

        # X: tap index (bottom) + delay ns (top)
        tap_tick_step = max(1, NUM_CIR_TAPS // 8)
        tap_ticks = np.arange(0, NUM_CIR_TAPS, tap_tick_step)
        ax2.set_xticks(tap_ticks)
        ax2.set_xticklabels([f"{int(c)}" for c in tap_ticks], fontsize=6)
        ax2.set_xlabel("Delay Tap Index  (amplitude →)", fontsize=8)
        ax2_top = ax2.twiny()
        ax2_top.set_xlim(ax2.get_xlim())
        ax2_top.set_xticks(tap_ticks)
        ax2_top.set_xticklabels([f"{delay_ns[c]:.1f}" for c in tap_ticks], fontsize=6)
        ax2_top.set_xlabel("Delay τ (ns)", fontsize=8)

        # Y: frame index (left) + slow time µs (right)
        ax2.set_yticks(tick_rows)
        ax2.set_yticklabels(
            [f"{all_frame_indices[start + r]}" for r in tick_rows], fontsize=6)
        ax2.set_ylabel("Frame Index  (slow time ↑)", fontsize=8)
        ax2r = ax2.twinx()
        ax2r.set_ylim(ax2.get_ylim())
        ax2r.set_yticks(tick_rows)
        ax2r.set_yticklabels([f"{slow_us[r]:.0f}" for r in tick_rows], fontsize=6)
        ax2r.set_ylabel("Slow Time (µs)", fontsize=8)

        plt.colorbar(im2, cax=cb2, label="|h| dB")

        progress = int(20 * end / N)
        fig.suptitle(
            f"Rolling window  frames {all_frame_indices[start]}–{all_frame_indices[end-1]}  "
            f"({slow_us[0]:.0f}–{slow_us[-1]:.0f} µs)  |  "
            f"({'█'*progress}{'░'*(20-progress)})",
            fontsize=9
        )

        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[:, :, :3]
        writer.append_data(img)

    writer.close()
    plt.close(fig)
    print(f"  Rolling video saved → {output_file}")


# =============================
# MAIN
# =============================
def main():
    # ── Discover PLY frames ───────────────────────────────────────────────────
    all_frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))
    all_paths  = [os.path.join(FRAMES_DIR, f) for f in all_frames]
    total      = len(all_paths)
    print(f"Found {total} frames in {FRAMES_DIR}")

    # ── Blender FPS ───────────────────────────────────────────────────────────
    print(f"\nEnter Blender animation FPS (e.g. 24, 30, 60) : ", end="")
    blender_fps = float(input().strip())
    assert blender_fps > 0, "FPS must be positive"
    print(f"  Δt_slow  = 1/{blender_fps:.0f} s = {1e6/blender_fps:.2f} µs / frame")
    print(f"  Δτ_delay = 1/BW = {DELTA_TAU_DELAY*1e9:.2f} ns / tap  (fixed by BW)")

    # ── Frames per spectrogram ────────────────────────────────────────────────
    print(f"\nEnter number of frames per spectrogram (min 2, max {total}) : ", end="")
    N_spec = int(input().strip())
    assert 2 <= N_spec <= total, f"Must be between 2 and {total}"

    # ── Start frame ───────────────────────────────────────────────────────────
    print(f"Enter start frame index (0-based, max {total - N_spec}) : ", end="")
    start_idx = int(input().strip())
    assert 0 <= start_idx <= total - N_spec, "Invalid start index"

    selected_paths   = all_paths[start_idx : start_idx + N_spec]
    selected_indices = list(range(start_idx, start_idx + N_spec))

    t_start_us = selected_indices[0]  / blender_fps * 1e6
    t_end_us   = selected_indices[-1] / blender_fps * 1e6
    print(f"\nSpectrogram covers frames {start_idx}–{start_idx+N_spec-1}  "
          f"→  {t_start_us:.2f} µs – {t_end_us:.2f} µs")

    # ── Process frames ────────────────────────────────────────────────────────
    H_mat, CIR_mat, slow_time_axis = collect_frames(
        selected_paths, selected_indices, blender_fps
    )
    # H_mat, CIR_mat now shape: (N_spec, NUM_SUBCARRIERS / NUM_CIR_TAPS)

    # ── Save raw data ─────────────────────────────────────────────────────────
    out_npz = f"spectro_frames{start_idx}-{start_idx+N_spec-1}.npz"
    np.savez(
        out_npz,
        H_mat          = H_mat,
        CIR_mat        = CIR_mat,
        slow_time_axis = slow_time_axis,
        freq_axis      = np.linspace(-BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS),
        delay_axis_ns  = np.arange(NUM_CIR_TAPS) * DELTA_TAU_DELAY * 1e9,
        blender_fps    = blender_fps,
        bandwidth_hz   = BANDWIDTH,
    )
    print(f"\nRaw matrices saved → {out_npz}")

    # ── Plot & save spectrogram ───────────────────────────────────────────────
    out_png = f"spectrogram_frames{start_idx}-{start_idx+N_spec-1}.png"
    plot_spectrograms(
        H_mat, CIR_mat, slow_time_axis,
        blender_fps   = blender_fps,
        frame_indices = selected_indices,
        title_prefix  = f"Frames {start_idx}–{start_idx+N_spec-1}  |  ",
        save_path     = out_png,
        show          = True,
    )

    # ── Optional rolling-window video ─────────────────────────────────────────
    print("\nGenerate rolling-window spectrogram video over ALL frames? [y/N] : ", end="")
    if input().strip().lower() == "y":

        print(f"Window size in frames [default={N_spec}] : ", end="")
        w_in = input().strip()
        win  = int(w_in) if w_in else N_spec

        # Cache already-processed rows
        all_H   = [None] * total
        all_CIR = [None] * total
        for row, fi in enumerate(selected_indices):
            all_H[fi]   = H_mat[row, :]
            all_CIR[fi] = CIR_mat[row, :]

        # Process remaining frames
        remaining = sum(1 for x in all_H if x is None)
        if remaining:
            print(f"\nProcessing remaining {remaining} frames ...")
            for fi, path in enumerate(all_paths):
                if all_H[fi] is None:
                    H_i, CIR_i = process_frame(path, fi)
                    all_H[fi]  = H_i
                    all_CIR[fi] = CIR_i
                    print(f"  [{fi+1}/{total}] {all_frames[fi]}")

        generate_spectrogram_video(
            all_H, all_CIR,
            all_frame_indices = list(range(total)),
            blender_fps       = blender_fps,
            window_size       = win,
            output_file       = "spectrogram_rolling.mp4",
            fps               = 5,
        )


if __name__ == "__main__":
    main()