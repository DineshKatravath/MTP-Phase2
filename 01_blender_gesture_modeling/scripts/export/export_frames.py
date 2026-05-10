import bpy
import os

MESH_OBJECT_NAME = "Plane005.001"
OUTPUT_DIR = "/Users/dinesh/Documents/mtp/hand_models/no_movement/hand_frames"

scene = bpy.context.scene
os.makedirs(OUTPUT_DIR, exist_ok=True)

orig_obj = bpy.data.objects.get(MESH_OBJECT_NAME)
depsgraph = bpy.context.evaluated_depsgraph_get()

print("Starting export...")

for frame in range(scene.frame_start, scene.frame_end + 1):

    scene.frame_set(frame)

    eval_obj = orig_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    filepath = os.path.join(OUTPUT_DIR, f"hand_{frame:04d}.ply")

    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(mesh.vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(mesh.polygons)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # vertices
        for v in mesh.vertices:
            co = eval_obj.matrix_world @ v.co
            f.write(f"{co.x} {co.y} {co.z}\n")

        # faces (triangulated automatically by Blender)
        for p in mesh.polygons:
            verts = " ".join(str(v) for v in p.vertices)
            f.write(f"{len(p.vertices)} {verts}\n")

    eval_obj.to_mesh_clear()

    print(f"Exported frame {frame}")

print("Export complete.")