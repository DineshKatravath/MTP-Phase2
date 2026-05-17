# Blender Rigging and Animation — Problems and Solutions

This document is a companion to the main [README.md](README.md) in this module. It records the practical problems encountered while rigging and animating the hand model in Blender, the solutions that worked, and the operational quirks that future users of this pipeline need to be aware of — especially when driving the rig live from MediaPipe.

It is written as a working log rather than a polished tutorial, so that anyone re-doing this work later can recognise the same failure modes quickly instead of re-discovering them.

---

## 1. Mesh Preparation Problems

### 1.1 Imported mesh had non-applied transforms

When the low-poly hand mesh was first imported into Blender, its Object properties showed non-zero rotation and non-unit scale. Any armature built on top of this would inherit those transforms, and bone roll values would behave unpredictably (rotating a finger bone on its local X axis would visibly tilt the finger sideways instead of curling it).

**Fix.** Selected the mesh in Object Mode and applied all transforms with `Ctrl+A → All Transforms`. After applying, the Object properties panel reads Rotation `(0,0,0)`, Scale `(1,1,1)`, Location at the origin. Only then was the armature built.

### 1.2 Duplicate mesh objects from the source file

The source `.blend` contained leftover duplicate hand objects from the original artist's scene. These caused the scripting API (`bpy.data.objects["Hand"]`) to pick the wrong object intermittently.

**Fix.** Deleted every duplicate from the Outliner and renamed the surviving mesh to a single, predictable name. The animation scripts in `scripts/animation/` rely on this exact object name.

### 1.3 Mesh origin not at the wrist

The mesh's object origin was somewhere inside the palm, not at the wrist joint. This made the wrist bone's pivot inconsistent with where MediaPipe reports landmark 0 (wrist), so live-driven wrist rotation appeared to swing the whole hand around the wrong axis.

**Fix.** Entered Edit Mode, selected the wrist vertices, and used `Object → Set Origin → Origin to 3D Cursor` after snapping the cursor to the wrist. The rig's wrist bone head was placed at the same point.

---

## 2. Armature and Bone Hierarchy Problems

### 2.1 Bone roll inconsistencies

After extruding the 20 bones along the finger chains, each bone's local axes pointed in slightly different directions. Identical Euler X rotations on two PIP bones would curl one finger and twist the other.

**Fix.** With the armature in Edit Mode, used `Armature → Bone Roll → Recalculate Roll → View Axis` (with the viewport aligned to the palm) to make every finger bone share the same local axis convention. After this, "curl" really is a uniform local X rotation across all PIP/DIP bones.

### 2.2 Parenting fingers to palm metacarpals, not directly to the wrist

The first version of the rig parented all five MCP bones directly to the wrist. This is anatomically wrong: fingers should fan out of metacarpals, which themselves cup with the palm. Without the metacarpal layer, the rig could not represent the "palm cup" required for `FIST` and `PINCH` gestures.

**Fix.** Inserted five `palm_*` bones (one per finger) between the wrist and each finger's `_01` bone. Each `palm_*` bone has 1 DOF (cup rotation). This brings the rig to its final 20 bones / 32 DOF layout described in `README.md`.

### 2.3 Mesh deformation artefacts after parenting

Initial `Ctrl+P → Armature Deform → With Automatic Weights` produced visible pinching between finger segments and on the back of the palm.

**Fix.**
1. Enabled `Weight Paint` mode and inspected each bone's vertex group.
2. Repainted the boundary between adjacent segments so vertices fall off smoothly rather than stepping abruptly.
3. For the palm, manually zeroed weights on the back of the hand for the finger `_01` bones — they had been picking up palm vertices via automatic weighting.

### 2.4 Joint constraints (Limit Rotation) silently rejected by scripts

Per-bone `Limit Rotation` constraints were configured to keep poses anatomically plausible. The first version of the animation scripts wrote bone angles that exceeded these limits — Blender silently clamped them, producing animations that were correct in code but visibly wrong on screen.

**Fix.** Centralised every joint's allowed range in `scripts/core/config.py` and made the gesture generators read those ranges before assigning angles. The constraints in the rig and the limits in `config.py` are now the same source of truth.

---

## 3. Animation Scripting Problems

### 3.1 Euler rotation mode mismatch

Some bones were created with their rotation mode set to Quaternion (Blender's default), and the animation scripts assigned Euler values to them. Blender accepted the assignment without error but the bone did not rotate as expected.

**Fix.** A small loop in `scripts/core/utils.py` sets every pose bone's `rotation_mode` to `'XYZ'` Euler before any pose is applied. The 32-DOF parameterisation assumes Euler angles in this exact order.

### 3.2 Keyframes ignored on first run

Inserting keyframes via `bone.keyframe_insert(data_path="rotation_euler")` on the first script run had no effect — the bones rotated for a moment, then snapped back. This happened because the active action did not yet exist.

**Fix.** The animation scripts now create an Action on the armature explicitly at start-up if one is not already present, and the scene end frame is set to match `HOLD_STEP × N + TRANSITION_STEP × N`. After this, keyframes persist correctly.

### 3.3 Transition frames polluting the training set

The 10,000-frame dataset cycles `HOLD → TRANSITION → HOLD` between gestures. Initially every frame was labelled with the destination gesture, which meant ~30% of training samples were mid-transition poses tagged as the final gesture. This hurt classifier accuracy noticeably.

**Fix.** Transition frames are now explicitly labelled `transition` in the per-frame JSON log and excluded from the classifier training set in `04_ml_classification/`. The 700 clean labelled frames per class quoted in the main README are post-exclusion counts.

### 3.4 Frame export overwriting previous runs

`export_frames.py` initially wrote to a fixed output directory, so a second run silently overwrote the first dataset.

**Fix.** Each export run now creates a timestamped subdirectory. Old runs are preserved until the user deletes them explicitly.

---

## 4. Reconstruction and Verification Problems

### 4.1 JSON replay not exactly recovering the animation

The first version of `scene_reconstruction.py` replayed bone angles but produced slightly different meshes from the original animation. The discrepancy was traced to the wrist global position not being recorded — only its rotation was.

**Fix.** The pose JSON now records `wrist_pos` (global location) alongside the 32-DOF angle vector. Replay is bit-exact in the bone hierarchy and visually identical in the rendered mesh.

### 4.2 21-keypoint reconstruction was ambiguous around the thumb

Reconstructing the pose from 21 world-space joint positions (the `21rep_reconstruction.py` route) gave good results for fingers but a wobbly thumb. The thumb's CMC joint has 3 DOF and is poorly constrained by 4 thumb landmarks alone.

**Fix (partial).** A dedicated thumb local frame is constructed from landmarks 1→2 before the curl axis is derived. This is the same trick later reused in `reciver_v4.py` for the live MediaPipe receiver.

---

## 5. MediaPipe Live Mode — Operational Warnings

When the rig is driven live by `06_mediapipe_integration/sender.py` and one of the `blender_receivers/reciver_v4.py` (or earlier) scripts, several quirks become important. These are not bugs in the code — they are consequences of the rig being driven from outside its natural keyframe pipeline.

### 5.1 NEVER save the `.blend` file while a live MediaPipe session is running

This is the single most important operational rule of the live demo.

While the receiver script is running, MediaPipe is continuously updating bone rotations and, in some cases, the global wrist orientation in real time. The hand orientation drifts away from its canonical rest pose as you move your physical hand. If you press `Ctrl+S` at this point, Blender persists the drifted orientation as the new baseline of the `.blend` file. The next time you open the project, the rest pose is wrong, gesture scripts targeting canonical bone angles will produce skewed hands, and any subsequent dataset generation will be silently corrupted.

**Rule.** Stop the receiver, close the live session, and verify the hand is back in its canonical rest pose **before** saving. If you are unsure whether you may have saved during a live session, do not save again — recover the rig from the previous Git-tracked `.blend` first.

### 5.2 Re-aligning the hand manually after a live session

Even if you do not save, a long live session can leave the hand visibly rotated away from its canonical orientation in the viewport (for example, palm facing sideways instead of facing the camera). For demos, dataset generation, or rendering, you may want to bring the hand back to its canonical pose.

This realignment has to be done **manually in Blender's UI**. There is no script for it, because the correct target orientation depends on what you are doing next (RF simulation expects a specific Tx/Rx-facing pose, while pure visualisation may not).

**Procedure for manual hand re-orientation.**

1. Stop the MediaPipe receiver (terminate the script in Blender's Scripting workspace).
2. Switch from Pose Mode back to **Object Mode** (`Tab` or use the mode dropdown at the top-left of the 3D viewport).
3. Select the hand object (or the armature root, depending on which carries the global orientation).
4. On the **left-hand toolbar** of the 3D viewport, click the **Rotate** tool — it is the icon that looks like two curved arrows forming a circle (below the Move tool and above the Scale tool). With the Rotate tool active, three coloured rotation rings appear around the selection.
5. Drag the appropriate ring (red = X, green = Y, blue = Z) to rotate the hand back toward its canonical orientation. Hold `Ctrl` while dragging to snap rotations to 5° increments, which is usually enough to land back on a clean axis.
6. If you want exact numeric control instead, open the `N` panel on the right side of the viewport and type the target rotation values directly into the Rotation fields.
7. Once the hand is back where you want it, optionally apply the rotation with `Ctrl+A → Rotation` so that the new orientation becomes the new baseline `(0,0,0)`.

Only **after** completing this realignment is it safe to save the `.blend` file again.

### 5.3 Why the drift happens

MediaPipe reports landmarks in image-space normalised coordinates and the receiver maps them onto bone rotations via the palm coordinate frame. Small frame-to-frame errors in landmark localisation accumulate into the wrist rotation channel because the wrist has no upstream parent constraint — there is nothing to absorb the drift. The EMA smoothing in `reciver_v4.py` reduces high-frequency jitter but does not remove low-frequency drift.

This is a known limitation of the current receiver design. A future improvement would be to anchor the wrist orientation to a fixed reference (for instance, the first frame's palm normal) and apply only **relative** rotations from MediaPipe onward.

### 5.4 Safe workflow for live sessions

The following sequence keeps the rig clean:

1. Open the `.blend` file fresh from disk. Verify the hand is in its canonical rest pose.
2. Start the Blender receiver in the Scripting workspace.
3. Start the MediaPipe sender in a terminal.
4. Run the live demo for as long as needed.
5. **Stop the sender, then stop the receiver.** Do not close Blender yet.
6. Verify visually that the hand is still in (or close to) its canonical orientation.
7. If it has drifted, follow the manual realignment procedure in section 5.2.
8. Only then save — or, preferred, **do not save at all**: close Blender without saving and re-open from disk for the next session. The Blender project file is meant to be a stable, version-controlled artefact, not a live cache.

---

## 6. Summary of Recurring Lessons

A few patterns kept showing up across these problems and are worth internalising for anyone extending this work:

- **Apply every transform before rigging.** Almost every "the bones rotate wrong" bug traced back to non-applied scale or rotation on the mesh or armature object.
- **Make joint constraints and code share one source of truth.** The bone limits in the rig and the angle ranges in `scripts/core/config.py` must match, or one of them will silently override the other.
- **Treat the `.blend` file as immutable during live sessions.** All real-time drift belongs in memory only, never on disk.
- **Record the global wrist position, not just rotation.** Bone angles alone are not a complete pose — replay needs the wrist's world location too.
- **Always provide a manual recovery path.** Even with the cleanest scripts, live capture introduces drift that only a human can correct. The Object Mode + Rotate-tool procedure in section 5.2 is that recovery path.
