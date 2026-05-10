import numpy as np

data = np.load("linked/linked_all_frames.npz", allow_pickle=True)

H_mat           = data["H_mat"]
CIR_mat         = data["CIR_mat"]
world_positions = data["world_positions"]
joint_names     = data["joint_names"]
ply_indices     = data["ply_frame_indices"]
json_frames     = data["json_frame_numbers"]
rf_files        = data["rf_files"]
ply_files       = data["ply_files"]
gestures        = data["gestures"]

print("=" * 60)
print(f"Dataset summary")
print(f"  Total frames     : {H_mat.shape[0]}")
print(f"  Subcarriers      : {H_mat.shape[1]}")
print(f"  CIR taps         : {CIR_mat.shape[1]}")
print(f"  Joints           : {len(joint_names)}")
print(f"  Joint names      : {list(joint_names)}")
print("=" * 60)

wrist_col = list(joint_names).index("wrist")

for i in range(243,245):
    print(f"\n── Frame row {i} ──────────────────────────────────────")
    print(f"  PLY frame idx    : {ply_indices[i]}")
    print(f"  JSON frame no.   : {json_frames[i]}")
    print(f"  RF file          : {rf_files[i]}")
    print(f"  PLY file         : {ply_files[i]}")
    print(f"  Gesture          : {gestures[i]}")

    # CSI
    H_row = H_mat[i]
    print(f"  CSI  |H| mean    : {np.abs(H_row).mean():.4e}")
    print(f"  CSI  |H| max     : {np.abs(H_row).max():.4e}")
    print(f"  CSI  first 5     : {np.abs(H_row[:5]).round(6)}")

    # CIR
    CIR_row = CIR_mat[i]
    print(f"  CIR  |h| mean    : {np.abs(CIR_row).mean():.4e}")
    print(f"  CIR  |h| max     : {np.abs(CIR_row).max():.4e}")
    print(f"  CIR  first 5     : {np.abs(CIR_row[:5]).round(6)}")

    # Keypoints
    print(f"  Keypoints (world_pos) :")
    for j, jname in enumerate(joint_names):
        wp = world_positions[i, j]
        print(f"    {jname:<20s} : [{wp[0]:8.3f}, {wp[1]:8.3f}, {wp[2]:8.3f}]")