import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FRAMES_DIR = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_normalized"

frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))

# ── Check first, middle and last frame bounding boxes ────────────────────────
check_frames = [frames[0], frames[len(frames)//2], frames[-1]]

print("="*65)
print(f"{'Frame':<25} {'Min XYZ':>20} {'Max XYZ':>20} {'Center':>18}")
print("="*65)

for fname in check_frames:
    mesh  = trimesh.load(os.path.join(FRAMES_DIR, fname), process=False)
    verts = np.array(mesh.vertices)
    vmin  = verts.min(axis=0)
    vmax  = verts.max(axis=0)
    ctr   = verts.mean(axis=0)
    print(f"{fname:<25} {str(vmin.round(3)):>20} {str(vmax.round(3)):>20} {str(ctr.round(3)):>18}")

print("="*65)

# ── Check ALL frames — are they all centered? ─────────────────────────────────
print("\nChecking all frames for centering consistency...")
all_centers = []
all_spans   = []
for fname in frames:
    mesh  = trimesh.load(os.path.join(FRAMES_DIR, fname), process=False)
    verts = np.array(mesh.vertices)
    all_centers.append(verts.mean(axis=0))
    all_spans.append(verts.max() - verts.min())

all_centers = np.array(all_centers)
all_spans   = np.array(all_spans)

print(f"\nCenter X range : {all_centers[:,0].min():.4f} to {all_centers[:,0].max():.4f}")
print(f"Center Y range : {all_centers[:,1].min():.4f} to {all_centers[:,1].max():.4f}")
print(f"Center Z range : {all_centers[:,2].min():.4f} to {all_centers[:,2].max():.4f}")
print(f"\nSpan range     : {all_spans.min():.4f} to {all_spans.max():.4f}")
print(f"Mean span      : {all_spans.mean():.4f}")


TX_POSITION = np.array([-0.149, -0.026, 0.031])
RX_POSITION = np.array([ 0.329, -0.026, 0.031])
hand_center = all_centers.mean(axis=0)
hand_span   = all_spans.mean()

print(f"\nTX position    : {TX_POSITION}")
print(f"RX position    : {RX_POSITION}")
print(f"Hand center    : {hand_center.round(4)}")
print(f"Hand span      : {hand_span:.4f} m")
print(f"TX-RX distance : {np.linalg.norm(TX_POSITION - RX_POSITION):.4f} m")

# Assessment
tx_rx_dist = np.linalg.norm(TX_POSITION - RX_POSITION)
if hand_span > tx_rx_dist:
    print(f"\n⚠️  Hand ({hand_span:.3f}m) is LARGER than TX-RX gap ({tx_rx_dist:.3f}m)!")
    print(f"   TX and RX are INSIDE the hand mesh — increase TX-RX separation")
elif hand_span > tx_rx_dist * 0.8:
    print(f"\n⚠️  Hand ({hand_span:.3f}m) fills {hand_span/tx_rx_dist*100:.0f}% of TX-RX gap")
    print(f"   TX/RX may be clipping the hand — consider increasing separation slightly")
else:
    print(f"\n✓  Hand ({hand_span:.3f}m) fits within TX-RX gap ({tx_rx_dist:.3f}m)")
    print(f"   Hand fills {hand_span/tx_rx_dist*100:.0f}% of TX-RX gap")

if np.abs(hand_center).max() > 0.01:
    print(f"\n⚠️  Hand is NOT centered at origin — center is {hand_center.round(4)}")
    print(f"   This means the hand is offset from the TX-RX midpoint")
else:
    print(f"\n✓  Hand is correctly centered at origin (midpoint of TX and RX)")

# ── Visual check ──────────────────────────────────────────────────────────────
# Load first, middle, last frame meshes for plotting
plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 6), facecolor='#0d1117')
fig.suptitle("Hand Placement Check — Normalized Frames",
             fontsize=14, fontweight='bold', color='white')

sample_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
sample_names   = ["Frame 1\n(OPEN)", "Frame 60\n(FIST)",
                   "Frame 120\n(WAVE)", "Frame 180\n(THUMBS_UP)", "Frame 240\n(OPEN end)"]

for plot_idx, (frame_idx, name) in enumerate(zip(sample_indices, sample_names)):
    ax = fig.add_subplot(1, 5, plot_idx+1, projection='3d')

    mesh  = trimesh.load(os.path.join(FRAMES_DIR, frames[frame_idx]), process=False)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Draw hand mesh
    step = max(1, len(faces) // 1000)
    tri  = Poly3DCollection(verts[faces[::step]],
                             alpha=0.4, facecolor='#D4956A',
                             edgecolor='#8B5E3C', linewidth=0.1)
    ax.add_collection3d(tri)

    # Draw TX and RX
    ax.scatter(*TX_POSITION, c='#FF4444', s=200, marker='^',
               edgecolors='white', linewidths=1, zorder=9)
    ax.scatter(*RX_POSITION, c='#44FF88', s=200, marker='v',
               edgecolors='white', linewidths=1, zorder=9)
    ax.text(TX_POSITION[0], TX_POSITION[1], TX_POSITION[2]+0.02,
            'TX', color='#FF4444', fontsize=8, fontweight='bold')
    ax.text(RX_POSITION[0], RX_POSITION[1], RX_POSITION[2]+0.02,
            'RX', color='#44FF88', fontsize=8, fontweight='bold')

    # Draw TX-RX line
    ax.plot([TX_POSITION[0], RX_POSITION[0]],
            [TX_POSITION[1], RX_POSITION[1]],
            [TX_POSITION[2], RX_POSITION[2]],
            '--', color='yellow', linewidth=1, alpha=0.5)

    # Axis limits centered on hand
    pad = 0.2
    all_pts = np.vstack([verts, TX_POSITION[None], RX_POSITION[None]])
    ax.set_xlim(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
    ax.set_ylim(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
    ax.set_zlim(all_pts[:,2].min()-pad, all_pts[:,2].max()+pad)

    ax.set_title(name, color='white', fontsize=9, pad=4)
    ax.set_xlabel("X", color='#aaaaaa', fontsize=7)
    ax.set_ylabel("Y", color='#aaaaaa', fontsize=7)
    ax.set_zlabel("Z", color='#aaaaaa', fontsize=7)
    ax.tick_params(colors='#aaaaaa', labelsize=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#2a2a2a')
    ax.yaxis.pane.set_edgecolor('#2a2a2a')
    ax.zaxis.pane.set_edgecolor('#2a2a2a')
    ax.set_facecolor('#161b22')
    ax.view_init(elev=20, azim=-60)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("placement_check.png", dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("\nSaved: placement_check.png")