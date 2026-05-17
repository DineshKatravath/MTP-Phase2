import os
import numpy as np
import trimesh

FRAMES_DIR = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_clean"
NORM_DIR   = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_normalized"
os.makedirs(NORM_DIR, exist_ok=True)

frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))

# step 1: finding  scale factor
# Loading frame 1 to determine the unit scale
ref_mesh  = trimesh.load(os.path.join(FRAMES_DIR, frames[0]), process=False)
ref_verts = np.array(ref_mesh.vertices)
ref_span  = ref_verts.max() - ref_verts.min()

# We want the hand to be ~0.20m in real scale (typical hand span)
scale_factor = 0.20 / ref_span

print(f"Reference span : {ref_span:.4f} Blender units")
print(f"Scale factor   : {scale_factor:.6f}  (1 unit = {scale_factor:.4f} m)")

# Check where the hand sits in Blender space
ref_center = ref_verts.mean(axis=0)
print(f"Frame 1 center : {ref_center.round(4)} Blender units")
print(f"Frame 1 center : {(ref_center * scale_factor).round(4)} meters after scaling")
print(f"\nThis is where the hand naturally sits in the scene.")
print(f"TX and RX should be placed relative to THIS position.\n")

# step 2: scale only, preserve natural position
print(f"Scaling {len(frames)} frames (no centering)...")
all_centers = []
all_spans   = []

for i, fname in enumerate(frames):
    mesh  = trimesh.load(os.path.join(FRAMES_DIR, fname), process=False)
    verts = np.array(mesh.vertices)

    # ONLY scale — preserve natural position
    verts_scaled = verts * scale_factor

    mesh.vertices = verts_scaled
    mesh.export(os.path.join(NORM_DIR, fname))

    c = verts_scaled.mean(axis=0)
    s = verts_scaled.max() - verts_scaled.min()
    all_centers.append(c)
    all_spans.append(s)

    if i % 20 == 0 or i == len(frames)-1:
        print(f"  [{i+1:3d}/{len(frames)}] {fname}  "
              f"center={c.round(3)}  span={s:.4f}m")

all_centers = np.array(all_centers)
all_spans   = np.array(all_spans)

# Step 3: getting TX/RX positions relative to hand's natural center and span
mean_center = all_centers.mean(axis=0)
mean_span   = all_spans.mean()

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"  Hand natural center : {mean_center.round(4)} m")
print(f"  Hand span range     : {all_spans.min():.4f} – {all_spans.max():.4f} m")
print(f"  Hand mean span      : {mean_span:.4f} m")
print(f"\nSUGGESTED TX/RX POSITIONS:")

# Place TX and RX on either side of the hand's natural center
# along the X axis, with enough clearance
max_span = all_spans.max()
clearance = max_span * 0.66 + 0.01 # 66% of max hand span + 1cm extra to ensure all frames fit comfortably

tx_x = mean_center[0] - clearance
rx_x = mean_center[0] + clearance

print(f"  TX_POSITION = [{tx_x:.3f}, {mean_center[1]:.3f}, {mean_center[2]:.3f}]")
print(f"  RX_POSITION = [{rx_x:.3f}, {mean_center[1]:.3f}, {mean_center[2]:.3f}]")
print(f"  TX-RX separation = {rx_x - tx_x:.3f} m")
print(f"  Hand fills ~{mean_span/(rx_x-tx_x)*100:.0f}% of TX-RX gap")

# Step 4: check if any frame's hand goes outside TX-RX bounds
print(f"\nChecking all frames stay within TX-RX bounds...")
outside = 0
for i, (c, s) in enumerate(zip(all_centers, all_spans)):
    hand_min_x = c[0] - s/2
    hand_max_x = c[0] + s/2
    if hand_min_x < tx_x or hand_max_x > rx_x:
        outside += 1
        if outside <= 5:  # print first 5 only
            print(f"   Frame {i+1} ({frames[i]}): "
                  f"hand_x=[{hand_min_x:.3f}, {hand_max_x:.3f}] "
                  f"outside TX-RX=[{tx_x:.3f}, {rx_x:.3f}]")

if outside == 0:
    print(f"   All {len(frames)} frames stay within TX-RX bounds")
else:
    print(f"   {outside} frames extend outside TX-RX bounds")
    print(f"   Increase clearance or adjust TX-RX positions")