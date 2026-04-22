# 05 — Blender to Sionna

## What This Is

The pipeline that takes Blender's animated hand mesh and prepares it for Sionna's ray-tracing engine. Three steps: export, clean, normalize.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `export_frames.py` | Export per-frame deformed mesh as PLY from Blender |
| `clean_frames.py` | Strip custom vertex attributes that crash Mitsuba |
| `preprocess_frames.py` | Scale to metric units, compute TX/RX positions |
| `hand_check.py` | Verify all frames are correctly scaled and positioned |
| `frame_preview.py` | Visualize hand + TX/RX + ray paths for the first frame |
| `frame_preview_v2.py` | Improved visualization — dark theme, normalized mesh |
| `sionna_setup.py` | First working Sionna scene (early version, used OBJ — now superseded) |

---

## Step 1 — Export PLY Frames

`export_frames.py` runs inside Blender. For each frame in the animation:
1. Sets the scene frame
2. Evaluates the deformed mesh (armature applied) via `depsgraph`
3. Creates a temporary object with the evaluated mesh
4. Triangulates faces (required for PLY export and Mitsuba)
5. Exports as `hand_NNNN.ply` to `OUTPUT_DIR`

```python
MESH_OBJECT_NAME = "Plane005.001"   # the hand mesh object name
OUTPUT_DIR = "/path/to/hand_frames"
```

**Why PLY and not OBJ?**  
OBJ export creates a `.mtl` file. Mitsuba reads `.mtl` and creates a BSDF (visual material). Sionna then crashes because it expects a RadioMaterial, not a BSDF. PLY carries only geometry — no material info.

---

## Step 2 — Clean PLY Files

`clean_frames.py` reloads each PLY with trimesh and re-exports it:

```python
mesh = trimesh.load(input_path, process=False)
mesh.export(output_path)
```

This strips custom vertex attributes (UV maps, vertex colors, shape key data) that can cause Mitsuba's PLY loader to crash. Output goes to `hand_frames_clean/`.

---

## Step 3 — Normalize to Meters

`preprocess_frames.py` scales the meshes from Blender units to real-world meters:

1. Load the first frame to measure its bounding box span
2. Compute `scale_factor = 0.20 / ref_span` so hand width ≈ 0.20 m
3. Apply the same scale factor to ALL frames (only scale — never center)
4. Compute TX and RX positions relative to the hand's natural center

```
Reference span : X Blender units
Scale factor   : 0.20 / X
Hand center    : [cx, cy, cz] meters (natural position, not centered)
TX_POSITION    = [cx - 0.75*span, cy, cz]
RX_POSITION    = [cx + 0.75*span, cy, cz]
```

**Why not center at origin?**  
If each frame is centered, the hand's global motion (moving forward, backward) is lost. The TX/RX must stay fixed relative to the real position of the hand, so centering would destroy the spatial variation that makes different poses produce different RF signals.

For this project:
```
TX_POSITION = [-0.143,  0.020, -0.020]
RX_POSITION = [ 0.167,  0.020, -0.020]
TX-RX separation = 0.310 m
Hand span ≈ 0.20 m (fills ~65% of TX-RX gap)
```

---

## Step 4 — Verify

`hand_check.py` checks all normalized frames:
- Prints bounding box and center for first, middle, last frames
- Checks whether any frame extends outside TX-RX bounds
- Generates a visual plot of 5 sample frames with TX/RX markers

---

## Folder Structure After Processing

```
hand_models/
├── hand_frames/              ← raw PLY from Blender export
├── hand_frames_clean/        ← trimesh-cleaned PLY
└── hand_frames_normalized/   ← scaled to meters, final input for Sionna
```

---

## Sionna XML Scene Format

The XML passed to Sionna for each frame uses `itu-radio-material` as the material type (required by Sionna's parser) and PLY as the mesh format:

```xml
<scene version="2.1.0">
    <bsdf type="itu-radio-material" id="mat-hand-{idx}">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.08"/>
    </bsdf>
    <shape type="ply">
        <string name="filename" value="{mesh_path}"/>
        <boolean name="face_normals" value="true"/>
        <ref id="mat-hand-{idx}" name="bsdf"/>
    </shape>
    ... (floor and walls)
</scene>
```

The `concrete` type in the XML is just a placeholder — the actual skin EM properties are applied in Python after loading.

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `BSDF has no attribute add_object` | OBJ with .mtl loaded | Use PLY; strip materials |
| `unsupported XML element: radio_material` | Wrong Mitsuba version | Use mitsuba==3.7.1, drjit==1.2.0 |
| `NoneType has no attribute startswith` | Missing material ID in XML | Add unique `id=` to every `<bsdf>` |
| Hand not visible / wrong size | Scale not applied | Run preprocess_frames.py first |
| All frames look identical | Frames were centered | Re-export without centering |
