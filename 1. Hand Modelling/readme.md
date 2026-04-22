# 01 — Hand Modeling

## What This Is

This folder covers the Blender hand rig — the mesh, the bone structure, and how the rig is set up for animation and export.

---

## Mesh

- **Source:** Low-poly hand mesh downloaded from Sketchfab  
  https://sketchfab.com/3d-models/low-poly-hand-3d-model-19c9ac5c369a468a95f081a3cc2ad4ac
- **Type:** Low-poly mesh (~1000–3000 faces), suitable for real-time animation and fast Sionna ray-tracing
- **Blender object:** The mesh is parented to the armature named `Armature`

---

## Bone Structure (20 Bones Total)

```
Armature
├── wrist                    ← 3 DOF (X, Y, Z rotation)
│
├── palm_index               ← 1 DOF (cup/splay)
├── palm_middle              ← 1 DOF
├── palm_ring                ← 1 DOF
├── palm_pinky               ← 1 DOF
├── palm_thumb               ← 1 DOF
│
├── index_01  (MCP)          ← 2 DOF (curl X + spread Z)
│   ├── index_02  (PIP)      ← 1 DOF (curl X)
│   └── index_03  (DIP)      ← 1 DOF (curl X)
│
├── middle_01 (MCP)          ← 2 DOF
│   ├── middle_02 (PIP)      ← 1 DOF
│   └── middle_03 (DIP)      ← 1 DOF
│
├── ring_01   (MCP)          ← 2 DOF
│   ├── ring_02   (PIP)      ← 1 DOF
│   └── ring_03   (DIP)      ← 1 DOF
│
├── pinky_01  (MCP)          ← 2 DOF
│   ├── pinky_02  (PIP)      ← 1 DOF
│   └── pinky_03  (DIP)      ← 1 DOF
│
└── thumb_01  (CMC/MCP)      ← 3 DOF (X, Y, Z)
    └── thumb_02  (IP)       ← 1 DOF (curl X)
```

**Total DOF:** ~27  
(4 fingers × 4 DOF + thumb × 4 DOF + 5 palm × 1 DOF + wrist × 3 DOF)

---

## Axis Convention

| Axis | Meaning |
|------|---------|
| X (rotation_euler[0]) | Curl — fingers fold toward palm |
| Y (rotation_euler[1]) | Twist — bone rotates along its length |
| Z (rotation_euler[2]) | Spread — fingers fan apart or together |

---

## Spread Bias

Each finger has a natural inward arc bias on the Z axis to make the default resting pose look natural (fingers don't all point straight parallel):

```python
BIAS = {
    "index":  -4,   # degrees
    "middle": -2,
    "ring":    2,
    "pinky":   4
}
```

---

## Curl Weight Distribution

When a finger curls, the base joint curls most, mid less, tip least:

```python
CURL_WEIGHTS = [1.0, 0.7, 0.4]   # base, mid, tip
```

This gives a natural finger curl where the base drives the motion and the tip follows.

---

## Setting Up in Blender

1. Open the `.blend` file containing the hand mesh and armature
2. Select the `Armature` object
3. Enter Pose Mode (`Ctrl+Tab`)
4. All Python scripts can now be run from the Blender Text Editor (`Scripting` workspace)
5. The armature must be the active object for scripts to work correctly

---

## Notes

- The armature object's rotation mode should be set to `QUATERNION` for stable global orientation when animating full hand motion in 3D space
- Individual bone rotations use `XYZ` Euler for simplicity and predictable axis behavior
- The palm bones have a subtle cupping effect — rotating them on X causes the palm to curve slightly, making fist poses more realistic
