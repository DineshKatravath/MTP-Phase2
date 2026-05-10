"""
live_rf_plot.py
---------------
Live RF visualization for CSI + CIR.

"""

import os
import time
import numpy as np
import matplotlib

# macOS GUI backend
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt


# ---------------- CONFIG ----------------

RF_DIR = "/Users/dinesh/Documents/mtp/hand_models/live_rf_plot_hand/rf_output"

IDLE_TIMEOUT  = 30.0
POLL_INTERVAL = 0.1


# ---------------- MAIN ----------------

def main():
    os.makedirs(RF_DIR, exist_ok=True)
    print("[plotter] Watching:", RF_DIR)

    seen_files     = set()
    last_seen_time = None

    # ---- Setup plot window ----
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Create lines ONCE (important for performance)
    line1, = ax1.plot([], [], linewidth=1.5)
    line2, = ax2.plot([], [], linewidth=1.5)

    ax1.set_title("CSI Magnitude")
    ax1.set_xlabel("Subcarrier Index")
    ax1.set_ylabel("|H(f)|")
    ax1.grid(True, alpha=0.3)

    ax2.set_title("CIR Magnitude")
    ax2.set_xlabel("Delay Bin")
    ax2.set_ylabel("|h(τ)|")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show window immediately (important for macOS)
    plt.show(block=False)
    plt.pause(0.1)

    print("[plotter] Plot window opened.")

    # ---------------- LOOP ----------------

    while True:

        files = sorted([f for f in os.listdir(RF_DIR) if f.endswith(".npz")])

        # ---- No files yet ----
        if not files:
            plt.pause(POLL_INTERVAL)
            continue

        # ---- Find new files (process ALL, in order) ----
        new_files = [f for f in files if f not in seen_files]

        if not new_files:
            if last_seen_time and (time.time() - last_seen_time > IDLE_TIMEOUT):
                print(f"[plotter] No new frames for {IDLE_TIMEOUT}s → exiting.")
                break
            plt.pause(POLL_INTERVAL)
            continue

        # ---- Process every new frame in order ----
        for fname in new_files:
            path = os.path.join(RF_DIR, fname)
            print(f"[plotter] Loading: {fname}")
            try:
                data = np.load(path)
                H   = data["H_real"]   + 1j * data["H_imag"]
                CIR = data["CIR_real"] + 1j * data["CIR_imag"]
                seen_files.add(fname)
            except Exception as e:
                print(f"[plotter] Error reading {fname}: {e}")
                seen_files.add(fname)
                continue

            # ---- Update plot with this frame ----
            line1.set_data(range(len(H)), np.abs(H))
            line2.set_data(range(len(CIR)), np.abs(CIR))

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()

            frame_num = int(fname.split("_")[-1].split(".")[0])
            fig.suptitle(
                f"Live mmWave 28 GHz — Hand RF Signal  |  Frame {frame_num:04d}",
                fontsize=12,
            )

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            print(f"[plotter] Updated: {fname}")

        last_seen_time = time.time()

    # ---- Exit cleanly ----
    print("[plotter] Finished.")
    plt.ioff()
    plt.show()


# ---------------- ENTRY ----------------

if __name__ == "__main__":
    main()