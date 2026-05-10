import os
import time
import shutil

from config import *
from frame_utils import *
from rf_engine import RFEngine


def main():

    print("=== Sionna Live Optimized Pipeline ===")

    # reset output dirs
    for d in [CLEAN_DIR, NORM_DIR, RF_DIR, PROCESSED_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    os.makedirs(LIVE_DIR, exist_ok=True)

    calibrated = False
    rf_engine = None
    scale_factor = None

    idx = 0
    last_seen = None

    while True:

        fname = get_latest_frame()

        if fname is None:
            if last_seen and time.time() - last_seen > IDLE_TIMEOUT:
                print("No frames — exiting")
                break
            time.sleep(POLL_INTERVAL)
            continue

        path = f"{LIVE_DIR}/{fname}"

        print(f"[{idx:04d}] Processing {fname}")

        # ---- LOAD ONCE ----
        mesh = load_mesh(path)

        if not calibrated:
            scale_factor = compute_scale_factor(mesh)
            normalize_mesh(mesh, scale_factor)

            tx, rx = calibrate_tx_rx(mesh)

            rf_engine = RFEngine(tx, rx)
            calibrated = True

        center, span = normalize_mesh(mesh, scale_factor)

        # save normalized mesh
        norm_path = f"{NORM_DIR}/{fname}"
        mesh.export(norm_path)

        # ---- RF ----
        start = time.time()
        H, CIR = rf_engine.run(norm_path, idx)
        print(f"RF done in {time.time()-start:.2f}s")

        save_rf(H, CIR, idx)

        shutil.move(path, f"{PROCESSED_DIR}/{fname}")

        last_seen = time.time()
        idx += 1


if __name__ == "__main__":
    main()