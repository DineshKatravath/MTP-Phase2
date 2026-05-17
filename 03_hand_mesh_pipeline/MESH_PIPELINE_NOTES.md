# Hand Mesh Pipeline — Problems and Solutions

This document is a companion to the main [README.md](README.md) in this module. It records the practical problems encountered while cleaning and aligning the `.ply` mesh frames between Blender export and Sionna RF simulation, the fixes that worked, and the operational warnings worth remembering.

It is written as a working log rather than a polished tutorial, so future users can recognise the same failure modes quickly instead of re-discovering them.

---

## 1. Raw Export Problems

### 1.1 Duplicate vertices made the mesh non-watertight

Raw `.ply` files exported by Blender contained many duplicated vertices along seams between mesh segments. Sionna's ray tracer would either treat these duplicates as separate surfaces (introducing phantom internal scatterers) or fail to intersect the surface cleanly along a seam.

**Fix.** `clean_frames.py` merges vertices within a small tolerance threshold and writes a watertight mesh. Without this step the simulated CSI shows narrow spikes that have nothing to do with hand pose — they come from the duplicate-vertex artefacts.

### 1.2 Degenerate triangles caused PathSolver errors

A few frames contained zero-area triangles, usually where two finger segments touched. Sionna would raise a numerical error in the ray-triangle intersection routine and abort the whole batch.

**Fix.** `clean_frames.py` removes any triangle whose area is below a tolerance threshold before writing the cleaned file. The cost is negligible (a handful of triangles per frame) and the batch no longer aborts.

### 1.3 Surface normals were stale after cleaning

After merging vertices and removing faces, the per-vertex normals stored in the original `.ply` no longer matched the surface. Sionna uses normals for material reflection direction, so stale normals produced subtly wrong CSI even when the geometry looked fine in a viewer.

**Fix.** Normals are recomputed at the end of `clean_frames.py` from the updated face list. Reflection direction is now consistent with the actual surface.

### 1.4 Mesh scale varied between exports

Different exports produced different physical scales because the Blender mesh's applied transforms drifted between sessions. A 5 cm hand and a 50 cm hand produce very different CSI even at the same pose.

**Fix.** `01_blender_gesture_modeling/preprocess_frames.py` normalises every cleaned mesh to the same physical extent in metres before Sionna sees it. Output goes to `normalized_frames/`, which is the directory the RF stage actually consumes.

---

## 2. Frame Indexing and Alignment Problems

### 2.1 Blender frame counter did not start at zero

Blender's scene frame counter starts at 1 by default and can be offset further by the animation script. Naive code that assumed `.ply` index 0 corresponded to JSON entry 0 produced a one-off shift between pose labels and RF results.

**Fix.** `link_rf_to_blender.py` reads the frame index directly out of each `.ply` filename (e.g. `hand_0042.ply → 42`) and matches that index against the `frame` field in the pose JSON. There is no implicit alignment by row order anywhere in the pipeline.

### 2.2 Parallel Sionna workers finished frames out of order

When `02_rf_simulation/scripts/save_rf_parallel_v2.py` runs, workers complete frames in non-deterministic order. The resulting `.npz` array could be a permutation of frame indices, which broke alignment if you trusted positional order.

**Fix.** The RF output `.npz` always carries an explicit `frame_ids` array. `link_rf_to_blender.py` joins on this array rather than on row position. If `frame_ids` is missing from an older archive, the script refuses to run rather than producing a silently wrong alignment.

### 2.3 Transition frames present in JSON but missing from RF

Earlier versions of the RF generator skipped transition frames to save compute, but the Blender pose JSON still contained them. A simple zip-by-index alignment would crash or shift labels.

**Fix.** `link_rf_to_blender.py` uses a left join keyed on RF frame IDs — every RF row gets its matching pose, and JSON entries with no RF counterpart are silently dropped. Transition frames are then filtered out by label string at the ML stage, not here.

### 2.4 Mismatched gesture labels at hold/transition boundaries

The first labelling scheme labelled the first frame of a transition with the destination gesture, which placed visibly mid-transition poses into the destination class. Classification accuracy dropped without an obvious cause.

**Fix.** All transition frames are labelled `transition` and excluded from training. `link_rf_to_blender.py` writes the label exactly as the JSON records it; it does not infer.

---

## 3. Wrist-Frame Subset Problems

### 3.1 Wrist-rotation Case 3 needed a separate subset

For Case 3 of the ML experiments (static poses with fast wrist rotation), the full dataset contained too much variation in finger pose, washing out the wrist-only signal.

**Fix.** A dedicated `wrist_frames/` directory holds the wrist-rotation subset, prepared by re-running `clean_frames.py` on a Blender export where only the wrist bone was animated. Keeping this as a separate subset prevents accidental mixing with the main 10,000-frame dataset.

### 3.2 Wrist-frame indices clashed with main-dataset indices

When the wrist subset and the main dataset were processed in the same output directory, their `.ply` filenames overlapped. RF output got overwritten and the linker silently joined wrong rows.

**Fix.** The wrist subset has its own subdirectory tree (`wrist_frames/`, with parallel `raw_frames`, `cleaned_frames`, and RF outputs). Indices are namespaced by directory, not by suffix.

---

## 4. Storage and Regeneration Problems

### 4.1 `.ply` datasets blew up the Git repository

A single 10,000-frame export is several gigabytes. Committing this to Git made clones unworkable and slowed every operation.

**Fix.** `.ply` directories (`raw_frames/`, `cleaned_frames/`, `normalized_frames/`, `wrist_frames/`) are all listed in `.gitignore` as generated data. They are regenerated locally from the Blender export plus `clean_frames.py` + `preprocess_frames.py`.

### 4.2 Re-running `clean_frames.py` silently appended to old output

The first version of the cleaner wrote into the output directory without first emptying it. After re-running the cleaner the directory contained a mix of new and stale `.ply` files, and downstream RF output was inconsistent with the pose JSON.

**Fix.** `clean_frames.py` now empties the output directory before writing. If that behaviour is dangerous in a given workflow, pass `--no-clear` and accept the risk explicitly.

### 4.3 Linked archive accidentally regenerated from stale RF

Running the full pipeline manually it is easy to skip the RF regeneration step but re-run `link_rf_to_blender.py`, producing an aligned archive that mixes a fresh pose JSON with a stale `.npz`.

**Fix (procedural).** `link_rf_to_blender.py` records the source RF archive's modification time in the output `.npz` metadata. ML scripts in `04_ml_classification/` warn if the source archive is older than the pose JSON, prompting a regeneration.

---

## 5. Validation Problems

### 5.1 Cleaned mesh visually fine but RF still wrong

Several times a cleaned mesh looked correct in MeshLab but Sionna still gave nonsense CSI. The problem was usually flipped normals on a small region of the palm.

**Fix.** Add a normal-orientation sanity check to `clean_frames.py`: after recomputing normals, assert that the majority point outward (using a ray-cast or a simple "normals agree with centroid-to-vertex direction" test). Frames that fail are flagged for manual inspection.

### 5.2 Hard to tell whether the link was correct

Once the linked `.npz` was produced there was no obvious way to verify that frame N's CSI really matched frame N's pose. Mis-alignments were only discovered later, after ML training had wasted compute.

**Fix.** `link_rf_to_blender.py` writes a small text summary alongside the `.npz` listing the first and last frame ID, gesture distribution, and any frames that were dropped. A quick read of this summary catches most alignment problems before training.

---

## 6. Summary of Recurring Lessons

- **Always join on explicit frame IDs, never on row order.** Parallel workers, dropped transition frames, and Blender's non-zero frame counter all guarantee that row order will lie to you sooner or later.
- **Clean the geometry before tuning the simulator.** Most "wrong-looking" CSI investigations bottomed out in mesh problems (duplicates, degenerate faces, flipped normals), not in Sionna parameters.
- **Treat generated mesh directories as ephemeral.** They are never committed, are always reproducible from Blender + cleaner, and should be wiped before a fresh regeneration.
- **Validate alignment before training.** A two-line summary text file next to the linked `.npz` catches mistakes that hours of training cannot.
