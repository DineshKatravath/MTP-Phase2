import trimesh

def clean_ply(input_path, output_path):
    mesh = trimesh.load(input_path, process=False)
    # Exporting only clean geometry, no custom vertex attributes
    mesh.export(output_path)

# Pre-processing all frames once
import os
# change these paths to your actual directories
FRAMES_DIR = "/Users/dinesh/Documents/mtp/hand_models/hand_frames"
CLEAN_DIR = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

for f in sorted(os.listdir(FRAMES_DIR)):
    if f.endswith(".ply"):
        clean_ply(
            os.path.join(FRAMES_DIR, f),
            os.path.join(CLEAN_DIR, f)
        )