import os
import numpy as np
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver, RadioMaterial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# CONFIGURATION

FRAMES_DIR   = "/Users/dinesh/Documents/mtp/hand_models/with_wrist_movement/hand_frames_normalized"
FREQUENCY    = 28e9
MAX_DEPTH    = 6
TX_POSITION = np.array([-0.113, -0.006, -0.002])
RX_POSITION = np.array([0.208, -0.006, -0.002])
subcarrier_freqs = (FREQUENCY + np.linspace(-200e6, 200e6, 256)).astype(np.float32)

frames    = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".ply"))
mesh_path = os.path.join(FRAMES_DIR, frames[0])

# Loading normalized mesh 
hand_mesh  = trimesh.load(mesh_path, process=False)
hand_verts = np.array(hand_mesh.vertices)
hand_faces = np.array(hand_mesh.faces)

print(f"Mesh loaded: {len(hand_verts)} vertices, {len(hand_faces)} faces")
print(f"Bounding box: {hand_verts.min(axis=0).round(4)} → {hand_verts.max(axis=0).round(4)}")
print(f"Center      : {hand_verts.mean(axis=0).round(4)}")
print(f"TX at       : {TX_POSITION}")
print(f"RX at       : {RX_POSITION}")

# Sionna scene 
xml = f"""<scene version="2.1.0">
<bsdf type="itu-radio-material" id="mat-hand">
    <string name="type" value="concrete"/>
    <float name="thickness" value="0.08"/>
</bsdf>
<shape type="ply">
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

# Overriding with skin EM properties
skin = RadioMaterial("skin_vis", relative_permittivity=17.3, conductivity=25.6)
for obj in scene.objects.values():
    obj.radio_material = skin

scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                              horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                              horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.add(Transmitter(name="tx", position=TX_POSITION))
scene.add(Receiver(name="rx",    position=RX_POSITION))

solver = PathSolver()
paths  = solver(scene=scene, max_depth=MAX_DEPTH, los=True,
                specular_reflection=True, diffuse_reflection=False,
                refraction=True, diffraction=False)

# Unpacking path vertices (already in same coordinate system as normalized mesh)
verts_raw = np.array(paths.vertices)[:, 0, 0, :, :]  # (depth, npaths, 3)
valid     = np.array(paths.valid)[0, 0, :]             # (npaths,)

tx_pos = np.array(TX_POSITION)
rx_pos = np.array(RX_POSITION)

path_colors = ['#2196F3', '#00BCD4', '#9C27B0', '#FF9800', '#E91E63']
path_labels = ['LOS', 'Reflection 1', 'Reflection 2', 'Refraction', 'Path 5']

ray_segs = []
for p in range(verts_raw.shape[1]):
    if not valid[p]:
        continue
    bp   = verts_raw[:, p, :]
    mask = ~np.all(bp == 0, axis=1)
    bp   = bp[mask]
    # NO normalization needed — mesh and paths are already in same space
    ray_segs.append(np.vstack([tx_pos, bp, rx_pos]))

print(f"\nValid propagation paths: {len(ray_segs)}")
for i, seg in enumerate(ray_segs):
    lbl = path_labels[i] if i < len(path_labels) else f'Path {i}'
    print(f"  {lbl}: {len(seg)-2} bounce points — {seg}")

# FIGURE

plt.style.use('dark_background')
fig = plt.figure(figsize=(20, 9), facecolor='#0d1117')
fig.suptitle(f"mmWave 28 GHz — TX / RX / Hand Placement\n{frames[0]}  |  Normalized Mesh",
             fontsize=15, fontweight='bold', color='white', y=0.98)

gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], projection='3d')
ax1.set_facecolor('#161b22')
fig.patch.set_facecolor('#0d1117')

tx_pos2 = np.array(TX_POSITION)
rx_pos2 = np.array(RX_POSITION)

def draw_txrx_2d(ax):
    ax.scatter(*tx_pos2[:2], c='#FF4444', s=400, zorder=9,
               marker='^', edgecolors='white', linewidths=1.5)
    ax.scatter(*rx_pos2[:2], c='#44FF88', s=400, zorder=9,
               marker='v', edgecolors='white', linewidths=1.5)
    ax.annotate('TX', tx_pos2[:2], xytext=(tx_pos2[0]-0.08, tx_pos2[1]+0.02),
                textcoords='data', color='#FF4444', fontsize=13, fontweight='bold')
    ax.annotate('RX', rx_pos2[:2], xytext=(rx_pos2[0]+0.02, rx_pos2[1]+0.02),
                textcoords='data', color='#44FF88', fontsize=13, fontweight='bold')

def draw_txrx_3d(ax):
    ax.scatter(*tx_pos2, c='#FF4444', s=400, zorder=9,
               marker='^', edgecolors='white', linewidths=1.5, label='TX')
    ax.scatter(*rx_pos2, c='#44FF88', s=400, zorder=9,
               marker='v', edgecolors='white', linewidths=1.5, label='RX')
    ax.text(tx_pos2[0]-0.1, tx_pos2[1], tx_pos2[2]+0.015,
            'TX', color='#FF4444', fontsize=12, fontweight='bold')
    ax.text(rx_pos2[0]+0.02, rx_pos2[1], rx_pos2[2]+0.015,
            'RX', color='#44FF88', fontsize=12, fontweight='bold')


# LEFT: 2D Top-Down

ax1.set_title("Top-Down View  (X-Y plane)", fontsize=13, color='white', pad=12)

ax1.tripcolor(hand_verts[:, 0], hand_verts[:, 1], hand_faces,
              facecolors=np.ones(len(hand_faces)),
              cmap='YlOrBr', alpha=0.55, edgecolors='none')
ax1.triplot(hand_verts[:, 0], hand_verts[:, 1], hand_faces,
            color='#8B6914', linewidth=0.25, alpha=0.5)

for i, seg in enumerate(ray_segs):
    c   = path_colors[i % len(path_colors)]
    lbl = path_labels[i] if i < len(path_labels) else f'Path {i}'
    ax1.plot(seg[:, 0], seg[:, 1], '-', color=c,
             linewidth=2.5, alpha=0.9, label=lbl, zorder=5)
    ax1.plot(seg[0,  0], seg[0,  1], 'o', color=c, markersize=7, zorder=6)
    ax1.plot(seg[-1, 0], seg[-1, 1], 'o', color=c, markersize=7, zorder=6)
    if len(seg) > 2:
        ax1.scatter(seg[1:-1, 0], seg[1:-1, 1],
                    color=c, s=100, marker='*', zorder=7)

draw_txrx_2d(ax1)

ax1.set_xlabel("X (m)", color='#aaaaaa', fontsize=11)
ax1.set_ylabel("Y (m)", color='#aaaaaa', fontsize=11)
ax1.tick_params(colors='#aaaaaa')
for spine in ax1.spines.values():
    spine.set_edgecolor('#333333')
ax1.grid(True, color='#222222', linewidth=0.6, alpha=0.8)
ax1.set_aspect('equal')
ax1.legend(loc='upper right', fontsize=9,
           facecolor='#1c2128', edgecolor='#444', labelcolor='white')

# RIGHT: 3D

ax2.set_title("3D View", fontsize=13, color='white', pad=12)

step      = max(1, len(hand_faces) // 5000)
s_faces   = hand_faces[::step]
tri_verts = hand_verts[s_faces]
poly = Poly3DCollection(tri_verts, alpha=0.55,
                        facecolor='#D4956A', edgecolor='#8B5E3C', linewidth=0.12)
ax2.add_collection3d(poly)

for i, seg in enumerate(ray_segs):
    c   = path_colors[i % len(path_colors)]
    lbl = path_labels[i] if i < len(path_labels) else f'Path {i}'
    ax2.plot(seg[:, 0], seg[:, 1], seg[:, 2], '-o', color=c,
             linewidth=3, markersize=7, alpha=0.95, label=lbl)
    if len(seg) > 2:
        ax2.scatter(seg[1:-1, 0], seg[1:-1, 1], seg[1:-1, 2],
                    color=c, s=100, marker='*', zorder=6)

draw_txrx_3d(ax2)

# Axis limits tight around everything
all_pts = np.vstack([hand_verts, tx_pos2[None], rx_pos2[None]] + list(ray_segs))
pad = 0.06
ax2.set_xlim(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
ax2.set_ylim(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
ax2.set_zlim(all_pts[:,2].min()-pad, all_pts[:,2].max()+pad)
ax2.set_xlabel("X (m)", color='#aaaaaa', fontsize=10, labelpad=8)
ax2.set_ylabel("Y (m)", color='#aaaaaa', fontsize=10, labelpad=8)
ax2.set_zlabel("Z (m)", color='#aaaaaa', fontsize=10, labelpad=8)
ax2.tick_params(colors='#aaaaaa')
ax2.xaxis.pane.fill = False
ax2.yaxis.pane.fill = False
ax2.zaxis.pane.fill = False
ax2.xaxis.pane.set_edgecolor('#2a2a2a')
ax2.yaxis.pane.set_edgecolor('#2a2a2a')
ax2.zaxis.pane.set_edgecolor('#2a2a2a')
ax2.grid(True, color='#2a2a2a', linewidth=0.5)
ax2.view_init(elev=22, azim=-55)
ax2.legend(loc='upper left', fontsize=9,
           facecolor='#1c2128', edgecolor='#444', labelcolor='white')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("scene_visualization.png", dpi=160, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: scene_visualization.png")