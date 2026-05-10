#!/usr/bin/env python3
"""
do_restructure.py — Reorganise mtp/ into a clean GitHub layout.
Run from the mtp/ root: python3 do_restructure.py
"""
import os
import shutil
import sys

ROOT = "/Users/dinesh/Documents/mtp"
os.chdir(ROOT)


def mkdirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
        print(f"  mkdir {p}")


def mv(src, dst):
    src_full = os.path.join(ROOT, src)
    dst_full = os.path.join(ROOT, dst)
    if not os.path.exists(src_full):
        print(f"  SKIP  {src}  (not found)")
        return
    dst_parent = os.path.dirname(dst_full)
    os.makedirs(dst_parent, exist_ok=True)
    shutil.move(src_full, dst_full)
    print(f"  MOVE  {src} → {dst}")


# ── 1. Create directory skeleton ─────────────────────────────
print("\n=== Creating directories ===")
mkdirs(
    "01_blender_gesture_modeling/assets",
    "02_rf_simulation/scripts",
    "02_rf_simulation/exploration_scripts",
    "02_rf_simulation/scene_configs",
    "02_rf_simulation/data/rf_output_case1",
    "02_rf_simulation/data/rf_output_case2",
    "02_rf_simulation/data/rf_output_case3",
    "03_hand_mesh_pipeline/raw_frames",
    "03_hand_mesh_pipeline/cleaned_frames",
    "03_hand_mesh_pipeline/normalized_frames",
    "03_hand_mesh_pipeline/wrist_frames",
    "04_ml_classification/scripts",
    "04_ml_classification/results/case1",
    "04_ml_classification/results/case2",
    "04_ml_classification/results/case3",
    "04_ml_classification/results/cnn",
    "04_ml_classification/results/cnn_attention",
    "04_ml_classification/results/real_imag",
    "04_ml_classification/results/temporal",
    "05_live_demo",
    "06_mediapipe_integration",
    "07_visualization/images",
    "07_visualization/videos",
    "07_visualization/spectrograms",
    "docs",
)

# ── 2. Blender gesture modeling ──────────────────────────────
print("\n=== 01 — Blender gesture modeling ===")
for f in ["frame_preview.py", "frame_preview_v2.py", "hand_check.py", "preprocess_frames.py"]:
    mv(f"hand_models/{f}", f"01_blender_gesture_modeling/{f}")

for f in [
    "hand-low-poly.zip",
    "low-poly-hand-3d-model.zip",
    "temp_scene.xml",
]:
    mv(f"hand_models/{f}", f"01_blender_gesture_modeling/assets/{f}")

mv("hand_models/source",   "01_blender_gesture_modeling/assets/source")
mv("hand_models/source 2", "01_blender_gesture_modeling/assets/source2")
mv("hand_models/linked",   "01_blender_gesture_modeling/assets/linked")

# ── 3. RF simulation ─────────────────────────────────────────
print("\n=== 02 — RF simulation ===")
for f in ["save_rf_data.py", "save_rf_data_parallel.py", "save_rf_parallel_v2.py",
          "sionna_setup.py", "load_npz.py"]:
    mv(f"hand_models/{f}", f"02_rf_simulation/scripts/{f}")

for f in ["csi_gen_empty_space.py", "csi_gen_floor.py",
          "csi_gen_predefined_skin.py", "csi_gen_walls_floor.py"]:
    mv(f"hand_models/{f}", f"02_rf_simulation/exploration_scripts/{f}")

# RF output data
mv("hand_models/rf_output",            "02_rf_simulation/data/rf_output_case1/rf_output")
mv("hand_models/rf_output_fast",       "02_rf_simulation/data/rf_output_case1/rf_output_fast")
mv("hand_models/rf_output_parallel",   "02_rf_simulation/data/rf_output_case2/rf_output_parallel")
mv("hand_models/rf_output_parallel_v2","02_rf_simulation/data/rf_output_case3/rf_output_parallel_v2")
mv("hand_models/results_case1.zip",    "02_rf_simulation/data/rf_output_case1/results_case1.zip")
mv("hand_models/results_case2.zip",    "02_rf_simulation/data/rf_output_case2/results_case2.zip")
mv("hand_models/results_case3.zip",    "02_rf_simulation/data/rf_output_case3/results_case3.zip")
mv("hand_models/no_movement",          "02_rf_simulation/data/no_movement")
mv("hand_models/with_wrist_movement",  "02_rf_simulation/data/with_wrist_movement")
mv("hand_models/mmwave_gesture_sample.npy", "02_rf_simulation/data/mmwave_gesture_sample.npy")
mv("hand_models/sionna_env",           "02_rf_simulation/sionna_env")

# ── 4. Hand mesh pipeline ─────────────────────────────────────
print("\n=== 03 — Hand mesh pipeline ===")
mv("hand_models/clean_frames.py",       "03_hand_mesh_pipeline/clean_frames.py")
mv("hand_models/link_rf_to_blender.py", "03_hand_mesh_pipeline/link_rf_to_blender.py")
mv("hand_models/hand_frames",           "03_hand_mesh_pipeline/raw_frames/hand_frames")
mv("hand_models/hand_frames_clean",     "03_hand_mesh_pipeline/cleaned_frames/hand_frames_clean")
mv("hand_models/hand_frames_normalized","03_hand_mesh_pipeline/normalized_frames/hand_frames_normalized")
mv("hand_models/hand_frames_wrist",     "03_hand_mesh_pipeline/wrist_frames/hand_frames_wrist")

# ── 5. ML classification ─────────────────────────────────────
print("\n=== 04 — ML classification ===")
for f in ["ml_model.py", "ml_model_1d_cnn.py", "ml_model_continuous.py",
          "ml_model_with_csi_phase.py", "spectrogram_genertor.py"]:
    mv(f"hand_models/{f}", f"04_ml_classification/scripts/{f}")

mv("hand_models/results",               "04_ml_classification/results/case1/results")
mv("hand_models/results_cnn",           "04_ml_classification/results/cnn/results_cnn")
mv("hand_models/results_cnn_attention", "04_ml_classification/results/cnn_attention/results_cnn_attention")
mv("hand_models/results_real_imag",     "04_ml_classification/results/real_imag/results_real_imag")
mv("hand_models/results_temporal",      "04_ml_classification/results/temporal/results_temporal")
mv("hand_models/spectro_frames0-5.npz",  "04_ml_classification/results/spectro_frames0-5.npz")
mv("hand_models/spectro_frames0-15.npz", "04_ml_classification/results/spectro_frames0-15.npz")

# ── 6. Live demo ──────────────────────────────────────────────
print("\n=== 05 — Live demo ===")
mv("hand_models/live_rf_plot_hand", "05_live_demo/live_rf_plot_hand")

# ── 7. MediaPipe integration ──────────────────────────────────
print("\n=== 06 — MediaPipe integration ===")
mv("hand_models/mediaPipe", "06_mediapipe_integration/mediaPipe")
mv("hand_models/sender.py", "06_mediapipe_integration/sender.py")

# ── 8. Visualization ─────────────────────────────────────────
print("\n=== 07 — Visualization ===")
images_from_pics = [
    "blender_edit_mode_count.png", "blender_limit_rotation.png",
    "blender_rigged_hand.png", "gesture_fist.png", "gesture_ok.png",
    "gesture_palm.png", "gesture_point.png", "gesture_thumbsdown.png",
    "gesture_thumbsup.png", "json_structure.png", "mediapipe_demo_palm.png",
    "mediapipe_demo_palm2.png", "mediapipe_demo_thumbsup.png",
    "mediapipe_landmarks.png", "mediapipe_terminal_connected.png",
    "sionna_csi_cir_plot.png", "sionna_csi_empty_space.png",
    "sionna_csi_floor_only.png", "sionna_csi_live_fullroom.png",
    "sionna_txrx_placement.png",
]
for f in images_from_pics:
    mv(f"pics/{f}", f"07_visualization/images/{f}")

mv("hand_models/placement_check.png",      "07_visualization/images/placement_check.png")
mv("hand_models/scene_visualization.png",  "07_visualization/images/scene_visualization.png")
mv("hand_models/spectrogram_frames0-5.png",  "07_visualization/spectrograms/spectrogram_frames0-5.png")
mv("hand_models/spectrogram_frames0-15.png", "07_visualization/spectrograms/spectrogram_frames0-15.png")

videos = [
    "csi_cir_empty_output.mp4", "csi_cir_floor_output.mp4",
    "csi_cir_hand_28GHz.mp4", "csi_cir_output.mp4",
    "csi_cir_walls_floor_output.mp4", "spectrogram_rolling.mp4",
]
for v in videos:
    mv(f"hand_models/{v}", f"07_visualization/videos/{v}")

# ── 9. Docs ───────────────────────────────────────────────────
print("\n=== docs ===")
mv("CS24M018_MTP2_V2.pdf", "docs/CS24M018_MTP2_V2.pdf")

# ── 10. Clean empty dirs ──────────────────────────────────────
print("\n=== Cleanup ===")
for d in ["hand_models", "pics"]:
    d_full = os.path.join(ROOT, d)
    if os.path.isdir(d_full):
        remaining = list(os.walk(d_full))
        if all(len(files) == 0 for _, _, files in remaining):
            shutil.rmtree(d_full)
            print(f"  RMDIR {d}/")
        else:
            print(f"  KEEP  {d}/  (still has files)")

print("\n✅  Restructuring complete.")
