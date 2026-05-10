import os
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

FRAMES_DIR = "/Users/dinesh/Documents/mtp/hand_models/hand_frames_clean"
FREQUENCY = 28e9
MAX_DEPTH = 6

subcarrier_freqs = (FREQUENCY + np.linspace(-50e6, 50e6, 128)).astype(np.float32)

frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))
mesh_path = os.path.join(FRAMES_DIR, frames[0])

# ── Load mesh and normalize to meter scale centered at origin ─────────────────
hand_mesh = trimesh.load(mesh_path, process=False)
hand_verts = np.array(hand_mesh.vertices)
hand_faces = np.array(hand_mesh.faces)

# Center the mesh
centroid = hand_verts.mean(axis=0)
hand_verts_centered = hand_verts - centroid

# Scale so the hand fits in roughly 0.2m (realistic hand width)
current_span = hand_verts_centered.max() - hand_verts_centered.min()
scale_factor = 0.2 / current_span
hand_verts_norm = hand_verts_centered * scale_factor

print(f"Original span: {current_span:.2f} units")
print(f"Scale factor applied: {scale_factor:.6f}")
print(f"Normalized hand spans: {hand_verts_norm.min(axis=0)} to {hand_verts_norm.max(axis=0)}")

# ── Place TX and RX on either side of the normalized hand ────────────────────
TX_POSITION = [-0.149, -0.026, 0.031]
RX_POSITION = [ 0.329, -0.026, 0.031]
tx_pos = np.array(TX_POSITION)
rx_pos = np.array(RX_POSITION)

# ── Sionna scene (uses original mesh coordinates) ────────────────────────────
xml = f"""<scene version="2.1.0">
<bsdf type="itu-radio-material" id="mat-hand">
    <string name="type" value="metal"/>
    <float name="thickness" value="0.004"/>
</bsdf>
<shape type="ply" id="hand">
    <string name="filename" value="{mesh_path}"/>
    <boolean name="face_normals" value="true"/>
    <ref id="mat-hand" name="bsdf"/>
</shape>
</scene>"""

xml_path = "temp_scene.xml"
with open(xml_path, "w") as f:
    f.write(xml)

scene = load_scene(xml_path)
scene.frequency = FREQUENCY
scene.objects["merged-shapes"].radio_material.relative_permittivity = 17.3
scene.objects["merged-shapes"].radio_material.conductivity = 25.6

scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                              horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                              horizontal_spacing=0.5, pattern="iso", polarization="V")

scene.add(Transmitter(name="tx", position=TX_POSITION))
scene.add(Receiver(name="rx",    position=RX_POSITION))

solver = PathSolver()
paths = solver(scene=scene, max_depth=MAX_DEPTH, los=True,
               specular_reflection=True, diffuse_reflection=False,
               refraction=True, diffraction=False)

# ── Unpack and normalize path vertices same way as mesh ──────────────────────
verts = np.array(paths.vertices)[:, 0, 0, :, :]  # (max_depth, num_paths, 3)
valid = np.array(paths.valid)[0, 0, :]            # (num_paths,)

ray_segments = []
for p in range(verts.shape[1]):
    if not valid[p]:
        continue
    bounce_pts = verts[:, p, :]
    mask = ~np.all(bounce_pts == 0, axis=1)
    bounce_pts = bounce_pts[mask]
    # Normalize bounce points same as mesh
    bounce_pts_norm = (bounce_pts - centroid) * scale_factor
    full_path = np.vstack([tx_pos, bounce_pts_norm, rx_pos])
    ray_segments.append(full_path)

print(f"\n{len(ray_segments)} valid paths")

# ── Plotting ──────────────────────────────────────────────────────────────────
path_colors  = ['royalblue', 'darkcyan', 'mediumpurple', 'orange', 'crimson']
path_labels  = ['LOS', 'Reflected 1', 'Reflected 2', 'Refracted', 'Path 5']

fig = plt.figure(figsize=(18, 8))
fig.suptitle(f"mmWave Scene — {frames[0]}", fontsize=14, fontweight='bold')

# ── LEFT: Top-down 2D ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(121)
ax1.set_title("Top-Down View (X-Y plane)", fontsize=12)

# Hand mesh filled projection
ax1.triplot(hand_verts_norm[:, 0], hand_verts_norm[:, 1],
            hand_faces, color='#d4a574', linewidth=0.3, alpha=0.4)
ax1.tripcolor(hand_verts_norm[:, 0], hand_verts_norm[:, 1],
              hand_faces, facecolors=np.ones(len(hand_faces)),
              cmap='copper', alpha=0.25, edgecolors='none')

# Ray paths
for i, seg in enumerate(ray_segments):
    c = path_colors[i % len(path_colors)]
    lbl = path_labels[i] if i < len(path_labels) else f'Path {i}'
    ax1.plot(seg[:, 0], seg[:, 1], '-', color=c,
             linewidth=2.5, alpha=0.9, label=lbl, zorder=4)
    ax1.plot(seg[:, 0], seg[:, 1], 'o', color=c,
             markersize=6, zorder=5)
    # Bounce points
    if len(seg) > 2:
        ax1.scatter(seg[1:-1, 0], seg[1:-1, 1], color=c,
                    s=100, marker='*', zorder=6, linewidths=1.5)

# TX / RX with large clear markers
ax1.scatter(*tx_pos[:2], c='red',   s=400, zorder=8, marker='^',
            edgecolors='darkred',   linewidths=1.5, label='TX')
ax1.scatter(*rx_pos[:2], c='lime',  s=400, zorder=8, marker='v',
            edgecolors='darkgreen', linewidths=1.5, label='RX')
ax1.annotate('TX', tx_pos[:2], fontsize=13, color='red',
             fontweight='bold', xytext=(-0.07, 0.03), textcoords='data')
ax1.annotate('RX', rx_pos[:2], fontsize=13, color='green',
             fontweight='bold', xytext=( 0.52, 0.03), textcoords='data')

ax1.set_xlabel("X (m)", fontsize=11)
ax1.set_ylabel("Y (m)", fontsize=11)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.25)
ax1.set_aspect('equal')

# ── RIGHT: 3D ────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("3D View", fontsize=12)

# Hand mesh — subsample faces, good alpha and skin color
step = max(1, len(hand_faces) // 3000)
sampled_faces = hand_faces[::step]
tri_verts = hand_verts_norm[sampled_faces]
poly = Poly3DCollection(tri_verts,
                        alpha=0.45,
                        facecolor='peachpuff',
                        edgecolor='sienna',
                        linewidth=0.15)
ax2.add_collection3d(poly)

# Ray paths
for i, seg in enumerate(ray_segments):
    c = path_colors[i % len(path_colors)]
    lbl = path_labels[i] if i < len(path_labels) else f'Path {i}'
    ax2.plot(seg[:, 0], seg[:, 1], seg[:, 2], '-o', color=c,
             linewidth=3, markersize=7, alpha=0.95, label=lbl)
    if len(seg) > 2:
        ax2.scatter(seg[1:-1, 0], seg[1:-1, 1], seg[1:-1, 2],
                    color=c, s=120, marker='*', zorder=6)

# TX / RX markers
ax2.scatter(*tx_pos, c='red',  s=400, marker='^', zorder=8,
            edgecolors='darkred', linewidths=1.5, label='TX')
ax2.scatter(*rx_pos, c='lime', s=400, marker='v', zorder=8,
            edgecolors='darkgreen', linewidths=1.5, label='RX')
ax2.text(tx_pos[0]-0.08, tx_pos[1], tx_pos[2]+0.03,
         'TX', color='red',   fontsize=12, fontweight='bold')
ax2.text(rx_pos[0]+0.02, rx_pos[1], rx_pos[2]+0.03,
         'RX', color='green', fontsize=12, fontweight='bold')

# Fit axes tightly around everything
all_pts = np.vstack([hand_verts_norm, tx_pos[None], rx_pos[None]] +
                    [s for s in ray_segments])
pad = 0.05
ax2.set_xlim(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
ax2.set_ylim(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
ax2.set_zlim(all_pts[:,2].min()-pad, all_pts[:,2].max()+pad)
ax2.set_xlabel("X (m)", fontsize=10)
ax2.set_ylabel("Y (m)", fontsize=10)
ax2.set_zlabel("Z (m)", fontsize=10)
ax2.legend(loc='upper left', fontsize=9)
ax2.view_init(elev=25, azim=-60)  # good default viewing angle

plt.tight_layout()
plt.savefig("scene_visualization.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: scene_visualization.png")