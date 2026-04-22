# 03 — Gesture Animation

## What This Is

Python scripts that run inside Blender's scripting environment to create, keyframe, and animate hand gestures on the rig.

---

## Scripts in This Folder

| Script | Purpose |
|--------|---------|
| `config.py` | Bone names, axis constants, curl weights, spread bias |
| `utils.py` | Helper functions: get armature, reset hand, rotate bone |
| `modules.py` | Loads config, utils, gestures as Blender text modules |
| `gestures.py` | High-level gesture functions: `fist()`, `open_hand()`, `point()`, `peace()`, `pinch()` |
| `temp.py` | Scratch file for calling gesture functions during development |
| `palm_fist_palm.py` | Simple palm→fist→palm animation (40 frames) |
| `gestures_with_anim.py` | Full gesture sequence with wrist rotation |
| `gestures_anim_global.py` | Gesture sequence + global hand translation + keypoint logging to JSON |
| `gestures_anim_global_random.py` | Continuous random full-DOF animation + keypoint logging |
| `gestures_anim_21rep.py` | Gesture sequence with 21-point keypoint extraction (MediaPipe-compatible) |
| `random_pose_anim.py` | Random pose animation with predefined gesture keypoints |
| `random_pose.py` | Apply one random pose (no animation) |

---

## How to Run in Blender

1. Open the `.blend` file with the hand rig
2. Go to the `Scripting` workspace
3. Open or paste the script in the Text Editor
4. Click `Run Script`

For the modular scripts (`gestures.py`, `config.py`, `utils.py`):
1. Add each file as a Blender text block (Text Editor → Open → select file)
2. Run `modules.py` first to register them as Python modules
3. Then run `temp.py` to call gestures

---

## Gesture Dictionary Format

Gestures are defined as dictionaries with these keys:

```python
gesture = {
    "fingers": {
        "index":  curl_degrees,   # 0 = straight, 90 = fully curled
        "middle": curl_degrees,
        "ring":   curl_degrees,
        "pinky":  curl_degrees,
    },
    "thumb":  curl_degrees,
    "spread": spread_degrees,     # finger fan spread
    "palm":   cup_degrees,        # palm cupping
    "wrist":  (rx, ry, rz),      # wrist Euler in degrees
}
```

### Built-in Gestures

| Name | Description |
|------|-------------|
| `OPEN` | All fingers straight, slight spread |
| `FIST` | All fingers fully curled |
| `THUMBS_UP` | Fingers curled, thumb pointing up |
| `THUMBS_DOWN` | Fingers curled, thumb pointing down |
| `WAVE_LEFT` | Open hand, wrist tilted left |
| `WAVE_RIGHT` | Open hand, wrist tilted right |

---

## Full-DOF Gesture Format (Advanced)

For the random pose generator and global animation scripts, a richer format is used:

```python
gesture = {
    "global_rot": (rx, ry, rz),    # whole arm rotation in degrees
    "wrist": (rx, ry, rz),
    "palm": {
        "palm_index": degrees,
        "palm_middle": degrees,
        "palm_ring": degrees,
        "palm_pinky": degrees,
        "palm_thumb": degrees,
    },
    "fingers": {
        "index": {
            "base": (mcp_curl, mcp_twist, mcp_spread),  # degrees
            "mid":  pip_degrees,
            "tip":  dip_degrees,
        },
        "middle": { ... },
        "ring":   { ... },
        "pinky":  { ... },
    },
    "thumb": {
        "base": (x_deg, y_deg, z_deg),
        "tip":  tip_deg,
    },
}
```

---

## Keyframing

All animation scripts use `keyframe_all()` which inserts keyframes on every bone at the current frame:

```python
def keyframe_all():
    for bone in pb:
        if bone.rotation_mode == 'QUATERNION':
            bone.keyframe_insert(data_path="rotation_quaternion")
        else:
            bone.keyframe_insert(data_path="rotation_euler")
    arm.keyframe_insert(data_path="location")
    arm.keyframe_insert(data_path="rotation_quaternion")
```

Blender then automatically interpolates between keyframes using its default bezier curves.

---

## Keypoint Export

After creating the animation, the scripts loop over every frame and extract joint data:

```python
for frame in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    # extract bone.head in world space for each bone
    # compute velocity = current_world_pos - prev_world_pos
```

Output: `hand_motion_full.json` (for global motion scripts) or `hand_keypoints_6dof.json` (for 21-rep scripts).

---

## Important Notes

- Always clear old animation data before creating a new one: `arm.animation_data_clear()`
- Set `arm.rotation_mode = 'QUATERNION'` on the armature object for stable global orientation
- Use `bpy.context.view_layer.update()` after `scene.frame_set()` to force Blender to recompute deformed positions
