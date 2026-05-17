# MediaPipe Integration — Problems and Solutions

This document is a companion to the main [README.md](README.md) in this module. It records the practical problems encountered while building the MediaPipe → Blender bridge for live hand capture, the fixes that worked, and the operational warnings worth remembering.

It is written as a working log rather than a polished tutorial, so future users can recognise the same failure modes quickly instead of re-discovering them.

---

## 1. MediaPipe Sender Problems

### 1.1 Wrong landmarker model variant

MediaPipe ships several hand-landmark model variants (`hand_landmarker.task` versus older `hands.tflite`). The first version of the sender used the older TFLite path, which returns only 2D coordinates. The Blender receiver expects 3D coordinates per landmark, so depth was always zero and curl-angle estimation collapsed.

**Fix.** Switched to the `HandLandmarker` task API with `hand_landmarker.task` and `num_hands=1`, which returns full 3D normalised coordinates per landmark. The model file is committed at `mediaPipe/hand_landmarker.task`.

### 1.2 Webcam grab blocked when no hand was in frame

Earlier sender code only sent a TCP message when MediaPipe detected a hand. If the user moved out of frame for a few seconds, the Blender receiver's socket starved and the bone smoothing snapped to a stale pose.

**Fix.** The sender now emits a heartbeat message (an empty landmark array) every webcam frame, regardless of detection. The receiver treats a heartbeat as "no update, keep the current pose" rather than re-applying the last detected pose at full intensity. This eliminates the snapping behaviour.

### 1.3 JSON serialisation overhead at high FPS

Encoding 21 × 3 floats as JSON every frame was a significant fraction of the sender's runtime budget at 30 fps. CPU usage spiked unexpectedly.

**Fix.** The JSON payload is built with a single `json.dumps()` call on a pre-allocated list rather than per-landmark concatenation. CPU usage on the sender side dropped to a level that comfortably fits inside a 30 fps webcam loop.

### 1.4 OpenCV camera index 0 was not the camera the user wanted

On laptops with both an internal and an external webcam, `cv2.VideoCapture(0)` could pick either depending on plug order. Users found the demo using the wrong camera with no obvious way to switch.

**Fix.** The camera index is read from an environment variable (or a `--camera` flag) at startup and printed clearly so the user can verify it. Defaults to 0, but is now easy to override.

---

## 2. TCP Transport Problems

### 2.1 Multiple JSON messages concatenated in a single `recv`

TCP is a stream, not a message protocol. A single `socket.recv(4096)` call sometimes returned the concatenation of two `json.dumps()` payloads, and `json.loads()` choked on the joined string.

**Fix.** Each JSON payload is now newline-terminated by the sender and the receiver reads with a buffered `recv` loop that splits on `\n` before parsing. Each line is one valid JSON message, end of story.

### 2.2 Receiver did not handle sender disconnects

If the sender exited mid-stream the receiver's `recv()` returned an empty bytes object, and the receiver crashed.

**Fix.** The receiver treats an empty `recv()` as "sender closed connection cleanly", returns the socket to a listening state, and waits for the next connection. The Blender script no longer needs to be restarted between sender invocations.

### 2.3 Connection refused on the first sender start

The sender used to start before the receiver was listening, so the first `connect()` failed and the sender exited immediately.

**Fix.** The sender retries `connect()` with a short backoff and exits only after several consecutive failures. In the live demo `run_pipeline.py` also enforces a start order (receiver first, then sender) so the first connection now almost always succeeds.

---

## 3. Blender Receiver Problems

### 3.1 Blocking socket I/O froze Blender's UI

Version 1 of the receiver used a blocking `socket.recv()` on the main thread. Blender's UI froze whenever the sender paused, and the user could not stop the script cleanly.

**Fix.** From v2 onward the receiver runs inside a `bpy.app.timers` callback at 10 ms intervals with a non-blocking socket. The UI remains responsive even if the sender stops sending.

### 3.2 Coarse curl estimation in v1

Receiver v1 set bone rotations to fixed angles based on a crude finger-open / finger-closed decision. The Blender hand could only really show two states per finger.

**Fix (v2).** A palm coordinate frame is constructed from landmark 0 (wrist) and the palm anchors at landmarks 5 and 17. Per-finger curl is computed by projecting the base-to-tip knuckle vector onto this palm plane, giving a continuous curl angle instead of an on/off state. The mapping is documented in §6 of the main `README.md`.

### 3.3 Twitchy armature from landmark jitter

Even with continuous angles the Blender hand twitched visibly because MediaPipe landmarks jitter from frame to frame.

**Fix (v3).** An exponential moving average (EMA) is applied to each bone's target angle. The smoothing window is configurable. v3 produces a visibly smoother live motion at the cost of a small added latency.

### 3.4 Thumb still wrong after smoothing

The smoothing in v3 helped fingers but the thumb was still wrong because curling the thumb does not move it in the palm plane — it moves out of it.

**Fix (v4).** A dedicated thumb local coordinate frame is constructed from landmarks 1 → 2 (the thumb metacarpal axis). Thumb curl is computed in this local frame rather than in the palm plane. The thumb now tracks correctly across all gesture classes.

### 3.5 Spread angle was implicitly zero

v1–v3 only drove curl, so the MCP lateral spread (`palm_*` Z rotation) stayed at zero. Gestures like `ROCK` and `V_SIGN` look qualitatively wrong without spread.

**Fix (v4).** Spread is estimated from the inter-finger base-landmark distance (e.g. landmark 5 vs 9 for index-vs-middle spread), normalised by hand size. This is enough to make `ROCK` and `V_SIGN` visibly correct.

---

## 4. Bone-Mapping Problems

### 4.1 Rotation mode mismatch

Some bones in the rig were left in Quaternion rotation mode by default. The receiver wrote Euler values to `rotation_euler`, which had no effect because the active rotation mode was Quaternion.

**Fix.** The receiver loops over every pose bone at startup and sets `rotation_mode = 'XYZ'` Euler. The bone hierarchy now matches what the receiver assumes.

### 4.2 Local axes inconsistent across fingers

After the bone-roll-recalculate step in the rig (see `01_blender_gesture_modeling/RIGGING_AND_ANIMATION_NOTES.md` §2.1), all finger bones share the same local axis convention. Without that step the receiver's "curl is local X rotation" assumption silently produced sideways finger motion on some fingers.

**Fix.** The receiver depends on the rig already having uniform bone-roll. If you import a different hand rig, redo the bone-roll recalculation first or the bone-mapping table will not apply.

### 4.3 Wrist rotation drift

The wrist bone has no upstream parent constraint, so small frame-to-frame errors in landmark localisation accumulate into the wrist rotation channel. EMA smoothing reduces high-frequency jitter but not low-frequency drift.

**Fix (partial).** Document the drift as a known limitation. Mitigate operationally: do not save the `.blend` file during a live session, and use the manual realignment procedure in §5 of the rigging notes after a long session.

A planned improvement is to anchor the wrist orientation to a fixed reference (the first frame's palm normal) and apply only relative rotations. This has not been implemented yet.

---

## 5. Operational Warnings

These are the same warnings already documented in `01_blender_gesture_modeling/RIGGING_AND_ANIMATION_NOTES.md` §5, repeated here because they are most likely to bite users of *this* module:

- **Never save the `.blend` file while a live MediaPipe session is running.** Hand orientation drifts in real time, and saving persists the drifted orientation as the new baseline. The next time you open the project, the rest pose will be wrong and all subsequent gesture scripts will produce skewed hands.
- **If you want to realign the hand after a live session**, you have to do it manually. There is no script for this:
  1. Stop the receiver (terminate the script in Blender's Scripting workspace).
  2. Switch from Pose Mode back to **Object Mode** (`Tab`).
  3. On the **left-hand toolbar** of the 3D viewport, click the **Rotate** tool (two curved arrows forming a circle, below Move and above Scale).
  4. Drag the coloured rotation rings (red = X, green = Y, blue = Z) to bring the hand back to its canonical orientation. Hold `Ctrl` to snap to 5° increments.
  5. Optionally apply the rotation with `Ctrl+A → Rotation` to lock in the new baseline.
  6. Only then save, or — preferred — close Blender without saving and reopen from disk for the next session.
- **Start the receiver before the sender.** The sender has retry logic but the cleanest behaviour is receiver-first.
- **Use one virtualenv per side.** Do not try to combine MediaPipe and Sionna in a single Python process; their dependency stacks collide.

---

## 6. Summary of Recurring Lessons

- **MediaPipe landmarks are noisy.** Plan smoothing in from day one — every receiver version added more smoothing or better local frames, none ever removed them.
- **Use local coordinate frames per anatomical part.** Palm frame for fingers, thumb metacarpal frame for the thumb. Forcing both through one global frame produced subtly wrong angles every time.
- **Newline-delimit JSON over TCP.** The "missing single-message guarantee" of TCP is the most common source of obscure parsing errors when streaming JSON.
- **The `.blend` file is a stable artefact.** Live state lives in memory, never on disk.
- **Document drift as a known limitation and give the user a recovery path.** Even the cleanest receiver design cannot fully prevent low-frequency drift; the Object-Mode + Rotate-tool procedure is the user-visible fallback.
