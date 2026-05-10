import os
import json
import numpy as np

# =============================
# CONFIGURATION
# =============================
FRAMES_DIR     = "/Users/dinesh/Documents/mtp/hand_models/no_movement/hand_frames_normalized"
KEYPOINTS_FILE = "/Users/dinesh/Documents/mtp/hand_models/source 2/hand_no_motion_dataset.json"
RF_OUTPUT_DIR  = "/Users/dinesh/Documents/mtp/hand_models/no_movement/rf_output_parallel"  
LINKED_OUT_DIR = "/Users/dinesh/Documents/mtp/hand_models/no_movement/linked"      


# =============================
# LOAD KEYPOINTS
# =============================
def load_keypoints(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {entry["frame"]: entry for entry in data}


# =============================
# LOAD ONE RF FRAME FILE
# =============================
def load_rf_frame(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "H"             : d["H_real"].astype(np.complex64) + 1j * d["H_imag"].astype(np.complex64),
        "CIR"           : d["CIR_real"].astype(np.complex64) + 1j * d["CIR_imag"].astype(np.complex64),
        "ply_frame_idx" : int(d["frame_idx"]),
        "json_frame"    : int(d["frame_idx"]),
        "freq_axis"     : d["freq_axis"],
    }


# =============================
# DISCOVER RF FRAME FILES
# =============================
def discover_rf_files(rf_dir, start_frame=None, end_frame=None):
    files = sorted(
        f for f in os.listdir(rf_dir)
        if f.startswith("rf_frame_") and f.endswith(".npz")
    )
    result = []
    for fname in files:
        idx = int(fname.replace("rf_frame_", "").replace(".npz", ""))
        if start_frame is not None and idx < start_frame:
            continue
        if end_frame is not None and idx > end_frame:
            continue
        result.append((idx, os.path.join(rf_dir, fname)))

    return result


# =============================
# BUILD LINKED DATASET
# =============================
def build_linked_dataset(rf_dir, keypoints_json_path, frames_dir,
                         start_frame=None, end_frame=None):

    rf_files = discover_rf_files(rf_dir, start_frame, end_frame)
    if not rf_files:
        raise FileNotFoundError(
            f"No rf_frame_NNN.npz files found in {rf_dir} "
            f"for range [{start_frame}, {end_frame}]"
        )
    print(f"\nFound {len(rf_files)} RF frame files  "
          f"(PLY idx {rf_files[0][0]} → {rf_files[-1][0]})")

    print(f"Loading keypoints : {keypoints_json_path}")
    kp_map = load_keypoints(keypoints_json_path)

    all_plys = sorted(f for f in os.listdir(frames_dir) if f.endswith(".ply"))

    linked      = []
    missing_kp  = []
    missing_ply = []

    for ply_idx, npz_path in rf_files:
        json_frame = ply_idx + 1

        rf = load_rf_frame(npz_path)

        ply_file = all_plys[ply_idx] if ply_idx < len(all_plys) else None
        kp_entry = kp_map.get(json_frame)

        if kp_entry is None:
            missing_kp.append(json_frame)
        if ply_file is None:
            missing_ply.append(ply_idx)

        linked.append({
            "ply_frame_idx"   : ply_idx,
            "json_frame"      : json_frame,
            "rf_file"         : os.path.basename(npz_path),
            "ply_file"        : ply_file,
            "H"               : rf["H"],
            "CIR"             : rf["CIR"],
            "freq_axis"       : rf["freq_axis"],
            "keypoints"       : kp_entry["joints"] if kp_entry else None,
            "global"          : kp_entry["global"] if kp_entry else None,
            "gesture"         : kp_entry.get("gesture", None) if kp_entry else None,
        })

    if missing_kp:
        print(f"  WARNING: {len(missing_kp)} frames missing keypoints : {missing_kp}")
    if missing_ply:
        print(f"  WARNING: {len(missing_ply)} frames missing PLY file  : {missing_ply}")
    if not missing_kp and not missing_ply:
        print(f"  All {len(linked)} frames linked successfully.")

    return linked


# =============================
# SAVE LINKED DATASET
# =============================
def save_linked(linked, out_path):
    N = len(linked)

    joint_names = []
    for e in linked:
        if e["keypoints"] is not None:
            joint_names = list(e["keypoints"].keys())
            break
    J = len(joint_names)

    H_mat   = np.stack([e["H"]   for e in linked]).astype(np.complex64)
    CIR_mat = np.stack([e["CIR"] for e in linked]).astype(np.complex64)

    world_pos = np.zeros((N, J, 3), dtype=np.float32)
    local_pos = np.zeros((N, J, 3), dtype=np.float32)
    rot_quat  = np.zeros((N, J, 4), dtype=np.float32)

    for i, entry in enumerate(linked):
        kp = entry["keypoints"]
        if kp is None:
            continue
        for j, jname in enumerate(joint_names):
            world_pos[i, j] = kp[jname]["world_position"]

            if "local_position" in kp[jname]:
                local_pos[i, j] = kp[jname]["local_position"]
            elif "world_position" in kp[jname]:
                local_pos[i, j] = kp[jname]["world_position"]
            else:
                raise KeyError(f"{jname} has neither 'local_position' nor 'world_position'")

            rot_quat[i, j]  = kp[jname]["rotation_quaternion"]

    ply_indices  = np.array([e["ply_frame_idx"] for e in linked], dtype=np.int32)
    json_frames  = np.array([e["json_frame"]    for e in linked], dtype=np.int32)
    rf_files_arr = np.array([e["rf_file"]  or "" for e in linked])
    ply_files_arr= np.array([e["ply_file"] or "" for e in linked])
    freq_axis    = linked[0]["freq_axis"]

    #  gestures
    gestures = np.array([e.get("gesture", "") for e in linked])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez(
        out_path,
        H_mat              = H_mat,
        CIR_mat            = CIR_mat,
        freq_axis          = freq_axis,
        world_positions    = world_pos,
        local_positions    = local_pos,
        rotations_quat     = rot_quat,
        joint_names        = joint_names,
        ply_frame_indices  = ply_indices,
        json_frame_numbers = json_frames,
        rf_files           = rf_files_arr,
        ply_files          = ply_files_arr,
        gestures           = gestures,
    )

    print(f"\nLinked dataset saved → {out_path}")
    print(f"  {N} frames  |  {J} joints  |  "
          f"{H_mat.shape[1]} subcarriers  |  {CIR_mat.shape[1]} CIR taps")
    print(f"  world_positions : {world_pos.shape}")
    print(f"  H_mat           : {H_mat.shape}  (complex64)")
    print(f"  CIR_mat         : {CIR_mat.shape}  (complex64)")


# =============================
# QUICK INSPECTION
# =============================
def inspect(linked, n=3):
    print(f"\n── First {n} linked entries ──")
    for e in linked[:n]:
        print(f"\n  PLY idx = {e['ply_frame_idx']:3d}  "
              f"JSON frame = {e['json_frame']:3d}  "
              f"rf_file = {e['rf_file']}  "
              f"ply_file = {e['ply_file']}")
        print(f"  H   : shape={e['H'].shape}   max|H|={np.abs(e['H']).max():.4e}")
        print(f"  CIR : shape={e['CIR'].shape}  max|h|={np.abs(e['CIR']).max():.4e}")
        if e["keypoints"]:
            wp = e["keypoints"]["wrist"]["world_position"]
            print(f"  wrist world_pos = {[round(v,3) for v in wp]}")
        else:
            print(f"  keypoints : MISSING")


# =============================
# MAIN
# =============================
def main():
    print(f"RF output dir     : {RF_OUTPUT_DIR}")
    print(f"Keypoints file    : {KEYPOINTS_FILE}")
    print(f"PLY frames dir    : {FRAMES_DIR}")

    print("\nLink ALL frames or a range? [all / range] : ", end="")
    choice = input().strip().lower()

    start_frame = None
    end_frame   = None

    if choice == "range":
        print("  Start PLY frame index (0-based) : ", end="")
        start_frame = int(input().strip())
        print("  End   PLY frame index (0-based) : ", end="")
        end_frame   = int(input().strip())
        tag = f"frames{start_frame}-{end_frame}"
    else:
        tag = "all_frames"

    linked = build_linked_dataset(
        rf_dir              = RF_OUTPUT_DIR,
        keypoints_json_path = KEYPOINTS_FILE,
        frames_dir          = FRAMES_DIR,
        start_frame         = start_frame,
        end_frame           = end_frame,
    )

    inspect(linked, n=3)

    out_path = os.path.join(LINKED_OUT_DIR, f"linked_{tag}.npz")
    save_linked(linked, out_path)


if __name__ == "__main__":
    main()

# =============================
# HOW TO RELOAD THE LINKED NPZ
# =============================
# data = np.load("linked/linked_all_frames.npz", allow_pickle=True)
#
# H_mat           = data["H_mat"]            # (N, 256) complex64
# CIR_mat         = data["CIR_mat"]          # (N, 256) complex64
# world_positions = data["world_positions"]  # (N, 20, 3) float32
# joint_names     = data["joint_names"]      # (20,) — name of each joint column
# ply_indices     = data["ply_frame_indices"]# (N,)  int32
# json_frames     = data["json_frame_numbers"] # (N,) int32
#
# # Example: get wrist world position for frame 5
# wrist_col = list(joint_names).index("wrist")
# wrist_pos = world_positions[5, wrist_col]  # (3,)