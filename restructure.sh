#!/usr/bin/env bash
# ============================================================
# restructure.sh — Reorganise mtp/ into a clean GitHub layout
# Run from the mtp/ root directory
# ============================================================
set -euo pipefail

ROOT="/Users/dinesh/Documents/mtp"
cd "$ROOT"

echo "=== Creating new top-level directories ==="
mkdir -p 01_blender_gesture_modeling/assets
mkdir -p 02_rf_simulation/scripts
mkdir -p 02_rf_simulation/exploration_scripts
mkdir -p 02_rf_simulation/scene_configs
mkdir -p 02_rf_simulation/data/rf_output_case1
mkdir -p 02_rf_simulation/data/rf_output_case2
mkdir -p 02_rf_simulation/data/rf_output_case3
mkdir -p 03_hand_mesh_pipeline/raw_frames
mkdir -p 03_hand_mesh_pipeline/cleaned_frames
mkdir -p 03_hand_mesh_pipeline/normalized_frames
mkdir -p 03_hand_mesh_pipeline/wrist_frames
mkdir -p 04_ml_classification/scripts
mkdir -p 04_ml_classification/results/case1
mkdir -p 04_ml_classification/results/case2
mkdir -p 04_ml_classification/results/case3
mkdir -p 04_ml_classification/results/cnn
mkdir -p 04_ml_classification/results/cnn_attention
mkdir -p 04_ml_classification/results/real_imag
mkdir -p 04_ml_classification/results/temporal
mkdir -p 05_live_demo
mkdir -p 06_mediapipe_integration
mkdir -p 07_visualization/images
mkdir -p 07_visualization/videos
mkdir -p 07_visualization/spectrograms
mkdir -p docs/assets
mkdir -p assets/hand_models_zip
echo "Directory tree created."

echo ""
echo "=== 01 — Blender gesture modeling ==="
# Blender Animations & raw 3D assets
mv "hand_models/gesture_dataset.py"       "01_blender_gesture_modeling/" 2>/dev/null || true
mv "hand_models/scene_reconstruction.py"  "01_blender_gesture_modeling/" 2>/dev/null || true
mv "hand_models/frame_preview.py"         "01_blender_gesture_modeling/" 2>/dev/null || true
mv "hand_models/frame_preview_v2.py"      "01_blender_gesture_modeling/" 2>/dev/null || true
mv "hand_models/hand_check.py"            "01_blender_gesture_modeling/" 2>/dev/null || true
mv "hand_models/preprocess_frames.py"     "01_blender_gesture_modeling/" 2>/dev/null || true
# Hand model 3-D assets
mv "hand_models/hand-low-poly.zip"        "01_blender_gesture_modeling/assets/" 2>/dev/null || true
mv "hand_models/low-poly-hand-3d-model.zip" "01_blender_gesture_modeling/assets/" 2>/dev/null || true
mv "hand_models/temp_scene.xml"           "01_blender_gesture_modeling/assets/" 2>/dev/null || true
mv "hand_models/source"                   "01_blender_gesture_modeling/assets/source" 2>/dev/null || true
mv "hand_models/source 2"                 "01_blender_gesture_modeling/assets/source2" 2>/dev/null || true
mv "hand_models/linked"                   "01_blender_gesture_modeling/assets/linked" 2>/dev/null || true

echo ""
echo "=== 02 — RF simulation ==="
mv "hand_models/save_rf_data.py"          "02_rf_simulation/scripts/" 2>/dev/null || true
mv "hand_models/save_rf_data_parallel.py" "02_rf_simulation/scripts/" 2>/dev/null || true
mv "hand_models/save_rf_parallel_v2.py"   "02_rf_simulation/scripts/" 2>/dev/null || true
mv "hand_models/sionna_setup.py"          "02_rf_simulation/scripts/" 2>/dev/null || true
mv "hand_models/csi_gen_empty_space.py"   "02_rf_simulation/exploration_scripts/" 2>/dev/null || true
mv "hand_models/csi_gen_floor.py"         "02_rf_simulation/exploration_scripts/" 2>/dev/null || true
mv "hand_models/csi_gen_predefined_skin.py" "02_rf_simulation/exploration_scripts/" 2>/dev/null || true
mv "hand_models/csi_gen_walls_floor.py"   "02_rf_simulation/exploration_scripts/" 2>/dev/null || true
mv "hand_models/load_npz.py"              "02_rf_simulation/scripts/" 2>/dev/null || true
# RF output data directories → sub-cases
mv "hand_models/rf_output"            "02_rf_simulation/data/rf_output_case1/rf_output" 2>/dev/null || true
mv "hand_models/rf_output_fast"       "02_rf_simulation/data/rf_output_case1/rf_output_fast" 2>/dev/null || true
mv "hand_models/rf_output_parallel"   "02_rf_simulation/data/rf_output_case2/rf_output_parallel" 2>/dev/null || true
mv "hand_models/rf_output_parallel_v2" "02_rf_simulation/data/rf_output_case3/rf_output_parallel_v2" 2>/dev/null || true
# Archived results zips
mv "hand_models/results_case1.zip"    "02_rf_simulation/data/rf_output_case1/" 2>/dev/null || true
mv "hand_models/results_case2.zip"    "02_rf_simulation/data/rf_output_case2/" 2>/dev/null || true
mv "hand_models/results_case3.zip"    "02_rf_simulation/data/rf_output_case3/" 2>/dev/null || true
# Scene-level no_movement / with_wrist reference captures
mv "hand_models/no_movement"          "02_rf_simulation/data/" 2>/dev/null || true
mv "hand_models/with_wrist_movement"  "02_rf_simulation/data/" 2>/dev/null || true
# mmWave sample
mv "hand_models/mmwave_gesture_sample.npy" "02_rf_simulation/data/" 2>/dev/null || true
# Sionna virtual env (large — git-ignored)
mv "hand_models/sionna_env"           "02_rf_simulation/" 2>/dev/null || true

echo ""
echo "=== 03 — Hand mesh pipeline ==="
mv "hand_models/clean_frames.py"         "03_hand_mesh_pipeline/" 2>/dev/null || true
mv "hand_models/link_rf_to_blender.py"   "03_hand_mesh_pipeline/" 2>/dev/null || true
mv "hand_models/hand_frames"             "03_hand_mesh_pipeline/raw_frames/hand_frames" 2>/dev/null || true
mv "hand_models/hand_frames_clean"       "03_hand_mesh_pipeline/cleaned_frames/hand_frames_clean" 2>/dev/null || true
mv "hand_models/hand_frames_normalized"  "03_hand_mesh_pipeline/normalized_frames/hand_frames_normalized" 2>/dev/null || true
mv "hand_models/hand_frames_wrist"       "03_hand_mesh_pipeline/wrist_frames/hand_frames_wrist" 2>/dev/null || true

echo ""
echo "=== 04 — ML classification ==="
mv "hand_models/ml_model.py"                 "04_ml_classification/scripts/" 2>/dev/null || true
mv "hand_models/ml_model_1d_cnn.py"          "04_ml_classification/scripts/" 2>/dev/null || true
mv "hand_models/ml_model_continuous.py"      "04_ml_classification/scripts/" 2>/dev/null || true
mv "hand_models/ml_model_with_csi_phase.py"  "04_ml_classification/scripts/" 2>/dev/null || true
mv "hand_models/spectrogram_genertor.py"     "04_ml_classification/scripts/" 2>/dev/null || true
# Results directories
mv "hand_models/results"               "04_ml_classification/results/case1/results" 2>/dev/null || true
mv "hand_models/results_cnn"           "04_ml_classification/results/cnn/results_cnn" 2>/dev/null || true
mv "hand_models/results_cnn_attention" "04_ml_classification/results/cnn_attention/results_cnn_attention" 2>/dev/null || true
mv "hand_models/results_real_imag"     "04_ml_classification/results/real_imag/results_real_imag" 2>/dev/null || true
mv "hand_models/results_temporal"      "04_ml_classification/results/temporal/results_temporal" 2>/dev/null || true
# Spectrogram data
mv "hand_models/spectro_frames0-5.npz"   "04_ml_classification/results/" 2>/dev/null || true
mv "hand_models/spectro_frames0-15.npz"  "04_ml_classification/results/" 2>/dev/null || true

echo ""
echo "=== 05 — Live demo ==="
# The whole live_rf_plot_hand folder becomes the live demo module
mv "hand_models/live_rf_plot_hand"  "05_live_demo/live_rf_plot_hand" 2>/dev/null || true

echo ""
echo "=== 06 — MediaPipe integration ==="
mv "hand_models/mediaPipe"            "06_mediapipe_integration/mediaPipe" 2>/dev/null || true
mv "hand_models/sender.py"            "06_mediapipe_integration/" 2>/dev/null || true

echo ""
echo "=== 07 — Visualization ==="
# Images from pics/
mv "pics/blender_edit_mode_count.png"     "07_visualization/images/" 2>/dev/null || true
mv "pics/blender_limit_rotation.png"      "07_visualization/images/" 2>/dev/null || true
mv "pics/blender_rigged_hand.png"         "07_visualization/images/" 2>/dev/null || true
mv "pics/gesture_fist.png"               "07_visualization/images/" 2>/dev/null || true
mv "pics/gesture_ok.png"                 "07_visualization/images/" 2>/dev/null || true
mv "pics/gesture_palm.png"               "07_visualization/images/" 2>/dev/null || true
mv "pics/gesture_point.png"              "07_visualization/images/" 2>/dev/null || true
mv "pics/gesture_thumbsdown.png"         "07_visualization/images/" 2>/dev/null || true
mv "pics/gesture_thumbsup.png"           "07_visualization/images/" 2>/dev/null || true
mv "pics/json_structure.png"             "07_visualization/images/" 2>/dev/null || true
mv "pics/mediapipe_demo_palm.png"        "07_visualization/images/" 2>/dev/null || true
mv "pics/mediapipe_demo_palm2.png"       "07_visualization/images/" 2>/dev/null || true
mv "pics/mediapipe_demo_thumbsup.png"    "07_visualization/images/" 2>/dev/null || true
mv "pics/mediapipe_landmarks.png"        "07_visualization/images/" 2>/dev/null || true
mv "pics/mediapipe_terminal_connected.png" "07_visualization/images/" 2>/dev/null || true
mv "pics/sionna_csi_cir_plot.png"        "07_visualization/images/" 2>/dev/null || true
mv "pics/sionna_csi_empty_space.png"     "07_visualization/images/" 2>/dev/null || true
mv "pics/sionna_csi_floor_only.png"      "07_visualization/images/" 2>/dev/null || true
mv "pics/sionna_csi_live_fullroom.png"   "07_visualization/images/" 2>/dev/null || true
mv "pics/sionna_txrx_placement.png"      "07_visualization/images/" 2>/dev/null || true
# Images scattered in hand_models/
mv "hand_models/placement_check.png"     "07_visualization/images/" 2>/dev/null || true
mv "hand_models/scene_visualization.png" "07_visualization/images/" 2>/dev/null || true
mv "hand_models/spectrogram_frames0-5.png"  "07_visualization/spectrograms/" 2>/dev/null || true
mv "hand_models/spectrogram_frames0-15.png" "07_visualization/spectrograms/" 2>/dev/null || true
# Videos
mv "hand_models/csi_cir_empty_output.mp4"     "07_visualization/videos/" 2>/dev/null || true
mv "hand_models/csi_cir_floor_output.mp4"     "07_visualization/videos/" 2>/dev/null || true
mv "hand_models/csi_cir_hand_28GHz.mp4"       "07_visualization/videos/" 2>/dev/null || true
mv "hand_models/csi_cir_output.mp4"           "07_visualization/videos/" 2>/dev/null || true
mv "hand_models/csi_cir_walls_floor_output.mp4" "07_visualization/videos/" 2>/dev/null || true
mv "hand_models/spectrogram_rolling.mp4"      "07_visualization/videos/" 2>/dev/null || true

echo ""
echo "=== docs ==="
mv "CS24M018_MTP2_V2.pdf"  "docs/" 2>/dev/null || true

echo ""
echo "=== Clean up now-empty hand_models/ and pics/ ==="
rmdir "hand_models" 2>/dev/null || echo "  hand_models/ not empty — leaving."
rmdir "pics"        2>/dev/null || echo "  pics/ not empty — leaving."

echo ""
echo "Done! Restructuring complete."
