import subprocess
import os
import threading
import time
import signal

BASE = "/Users/dinesh/Documents/mtp/05_live_demo/live_rf_plot_hand"

SENDER_PYTHON = "/Users/dinesh/Documents/mtp/06_mediapipe_integration/mediaPipe/mediaPipeEnv/bin/python"
SIONNA_PYTHON = "/Users/dinesh/Documents/mtp/02_rf_simulation/sionna_env/bin/python"

SENDER_SCRIPT  = os.path.join(BASE, "sender.py")
SIONNA_SCRIPT  = os.path.join(BASE, "sionna_live_parallel_rf.py")
PLOTTER_SCRIPT = os.path.join(BASE, "live_rf_plot.py")

HAND_DIR = os.path.join(BASE, "hand_frames_live")

def relay_output(proc, tag):
    for line in proc.stdout:
        print(f"[{tag}] {line}", end="")

def wait_for_frames():
    while True:
        if os.path.exists(HAND_DIR) and len(os.listdir(HAND_DIR)) > 0:
            print("[launcher] frames detected")
            return
        time.sleep(0.5)

def main():
    print("Starting pipeline...")

    # Start sionna
    sionna_proc = subprocess.Popen(
        [SIONNA_PYTHON, SIONNA_SCRIPT],
        cwd=BASE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    threading.Thread(target=relay_output, args=(sionna_proc, "sionna"), daemon=True).start()

    time.sleep(3)

    # Start sender
    sender_proc = subprocess.Popen(
        [SENDER_PYTHON, SENDER_SCRIPT],
        cwd=BASE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    threading.Thread(target=relay_output, args=(sender_proc, "sender"), daemon=True).start()

    # Wait until frames exist
    wait_for_frames()

    # Start plotter
    plotter_proc = subprocess.Popen(
        [SIONNA_PYTHON, PLOTTER_SCRIPT],
        cwd=BASE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    threading.Thread(target=relay_output, args=(plotter_proc, "plotter"), daemon=True).start()

    try:
        plotter_proc.wait()
    except KeyboardInterrupt:
        print("Stopping all...")

    for proc in [sender_proc, sionna_proc, plotter_proc]:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)

if __name__ == "__main__":
    main()