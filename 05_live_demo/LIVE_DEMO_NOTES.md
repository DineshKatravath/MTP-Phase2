# Live Demo — Problems and Solutions

This document is a companion to the main [README.md](README.md) in this module. It records the practical problems encountered while wiring together the live end-to-end demo (MediaPipe sender → Blender receiver → parallel Sionna worker pool → real-time CSI plot), the fixes that worked, and the operational warnings worth remembering.

It is written as a working log rather than a polished tutorial, so future users can recognise the same failure modes quickly instead of re-discovering them.

---

## 1. Orchestration Problems

### 1.1 Three processes with different Python environments

The pipeline has three independent processes:

- `sender.py` needs the MediaPipe environment (`mediapipe`, `opencv-python`).
- `sionna_live_parallel_rf.py` and `live_rf_plot.py` need the Sionna environment (Sionna, TensorFlow, Mitsuba).
- The Blender receiver runs inside Blender's embedded Python.

Trying to launch them from a single shell command failed because no one virtualenv contains all three sets of packages, and `bpy` does not exist outside Blender.

**Fix.** `run_pipeline.py` launches each external process with its own absolute-path Python interpreter via `subprocess.Popen`. The Blender receiver is **not** managed by `run_pipeline.py` — it has to be started manually from inside Blender's Scripting workspace before the pipeline is launched.

### 1.2 Start order mattered and was easy to get wrong

If the sender came up before the receiver, the TCP `connect()` failed and the sender exited. If the plotter came up before the Sionna worker pool, it sat blocked on an empty queue and the user thought the pipeline was hung.

**Fix.** `run_pipeline.py` enforces a strict start order:
1. Start the Sionna worker pool.
2. Wait 3 s for TensorFlow/Sionna to initialise.
3. Start the MediaPipe sender.
4. Wait until the first `.ply` file appears in `hand_frames_live/`.
5. Start the live CSI plotter.

The waits are explicit and the orchestrator prints `[tag]` prefixes for each subprocess so it is obvious which stage produced any given log line.

### 1.3 Ctrl+C leaked orphan processes

`SIGINT` to `run_pipeline.py` killed the orchestrator but the child processes — especially the Sionna worker pool — kept running and held the GPU.

**Fix.** `run_pipeline.py` installs a signal handler that sends `SIGINT` to every child, then `wait()`s on them with a short timeout, then `SIGKILL`s any survivors. The GPU is reliably released within a couple of seconds of Ctrl+C.

---

## 2. Blender Receiver Problems

### 2.1 Long-running socket reads froze Blender's UI

The first receiver used a blocking `socket.recv()` in the main thread. Blender's UI froze whenever the sender paused, and the user could not even close the script cleanly.

**Fix.** The production receiver runs the socket read inside a `bpy.app.timers` callback that fires every 10 ms. The socket is set non-blocking and the callback returns early if no data is available. This keeps Blender's UI responsive throughout the live session.

### 2.2 `.ply` export accumulated rapidly and filled the disk

At 10 ms per frame the receiver was writing thousands of `.ply` files per minute. Within an hour the project directory had eaten gigabytes of disk space.

**Fix.** The receiver exports at a lower rate (one frame every N timer ticks, configurable in `config.py`) so the Sionna worker pool can keep up without files accumulating. A separate cleanup routine in the orchestrator empties `hand_frames_live/` at startup and optionally at shutdown.

### 2.3 Drifted hand orientation persisted across runs

After a long live session the wrist orientation drifts away from the canonical rest pose (see `01_blender_gesture_modeling/RIGGING_AND_ANIMATION_NOTES.md` §5). If the user pressed `Ctrl+S` at any point, the next demo started with the wrong baseline orientation and every subsequent frame was skewed.

**Fix (procedural).** Do not save the `.blend` while a live session is running. To reset the orientation, follow the manual procedure documented in §5.2 of the rigging notes (Object Mode → Rotate tool → realign → optionally `Ctrl+A → Rotation`).

---

## 3. Frame-Watch and Dispatch Problems

### 3.1 Half-written `.ply` files dispatched to Sionna

The Sionna worker pool watched `hand_frames_live/` for new files. A `.ply` was sometimes picked up mid-write, before Blender had finished flushing. Sionna would crash with a malformed-mesh error.

**Fix.** The file-watch only submits a `.ply` once its file size has stayed constant across two consecutive 50 ms polls. This is a cheap proxy for "the write has finished" and eliminated the race.

### 3.2 Out-of-order CSI in the live plot

Workers finished frames at slightly different speeds. The plotter received CSI in completion order rather than frame order, producing visibly jumpy plots that did not match the user's hand motion.

**Fix.** The worker pool tags every CSI result with its frame ID. An in-order collector buffers results until the next-expected frame is available, then releases them to the plotter in strict order. Brief latency spikes from a single slow worker do not corrupt the plot.

### 3.3 Frame backlog grew unboundedly

If Sionna fell behind real time (which it does on slower hardware), the queue between the file-watch and the worker pool grew without bound, eventually exhausting memory.

**Fix.** The dispatch queue is capped, and old frames are dropped (not buffered) when the queue is full. The live demo is allowed to skip frames rather than fall further behind. The Sionna update rate is documented as 1–2 fps for this reason.

---

## 4. Sionna Worker Pool Problems

### 4.1 TensorFlow GPU OOM with multiple workers

Each Sionna worker imported TensorFlow and grabbed the full GPU memory on startup. With more than one worker the second process failed to initialise.

**Fix.** Every worker calls `tf.config.experimental.set_memory_growth(gpu, True)` before any TF or Sionna code runs. Pool size is tuned to GPU memory (typically 2–4 workers on a 12 GB GPU).

### 4.2 Workers re-built the Mitsuba scene on every frame

The first version rebuilt the room geometry from scratch for each frame. Scene construction was a significant fraction of per-frame cost.

**Fix.** Workers cache the room geometry at startup and only swap in the new hand mesh for each frame. `rf_scene.py` produces a parametric XML where only the hand `.ply` reference changes between frames.

### 4.3 Worker pickling errors with Sionna objects

Returning a Sionna PathSolver result directly through `multiprocessing` failed because some bound C++ objects could not be pickled.

**Fix.** Workers convert results to plain NumPy arrays (`H_mat`, `CIR_mat`) before returning. The pool transport carries only NumPy data, which pickles cleanly.

---

## 5. Live Plot Problems

### 5.1 Matplotlib animation flicker

The first plotter cleared and re-drew the whole figure on every update. The result flickered noticeably and felt sluggish.

**Fix.** `live_rf_plot.py` updates only the line data (`line.set_ydata(...)`) and calls `fig.canvas.draw_idle()` instead of redrawing the figure. The plot now updates smoothly at the Sionna update rate.

### 5.2 Axis rescaling on every frame hid the real signal

Matplotlib's auto-scaling re-fit the Y axis on every update. With a small variation in CSI magnitude the axis kept jumping, making the signal look constant even when it was changing.

**Fix.** The plot uses fixed Y-axis limits derived from a short calibration window at startup. The user sees actual variation rather than the auto-scaler's interpretation.

### 5.3 Plotter blocked the main loop in some Matplotlib backends

On some setups Matplotlib's `plt.show()` blocked, preventing the queue reader from running.

**Fix.** The plotter uses `plt.ion()` plus the `FuncAnimation` API explicitly, and the backend is selected non-interactively in `config.py`. The plot updates independently of the queue reader.

---

## 6. Environment and Path Problems

### 6.1 Absolute interpreter paths broke on other machines

`run_pipeline.py` references the MediaPipe and Sionna Python interpreters by absolute path. Cloning the repo onto a new machine made the orchestrator fail immediately because those paths did not exist.

**Fix (procedural).** `SENDER_PYTHON` and `SIONNA_PYTHON` are the two variables to update at the top of `run_pipeline.py`. They are deliberately not auto-detected — the explicit paths make environment confusion impossible once they are correct.

### 6.2 Mixing Python environments inside one process

A few early experiments tried to import MediaPipe and Sionna in the same Python process to avoid the multi-environment headache. The combined import order produced segfaults (CUDA + protobuf conflicts).

**Fix.** Accept the multi-environment design and keep MediaPipe and Sionna in fully separate virtualenvs. Communication between them is via the file system (`.ply` files) and pipes, never via shared Python state.

---

## 7. Operational Warnings

The following warnings are restatements of issues that have been seen to cost real time during live demos:

- **Never save the `.blend` file while a live session is running.** Hand orientation drifts during live capture, and saving persists the drifted orientation as the new baseline. See `01_blender_gesture_modeling/RIGGING_AND_ANIMATION_NOTES.md` §5 for the full reasoning and the recovery procedure.
- **If hand orientation looks wrong after a live session**, do not rerun the demo — first realign manually in Object Mode using the Rotate tool from the left-hand viewport toolbar, then continue.
- **Always start the Blender receiver first**, then run `run_pipeline.py`. The orchestrator cannot start Blender for you.
- **`hand_frames_live/` should be empty at startup.** Stale `.ply` files from a previous run will be picked up and replayed before the live frames arrive, producing a confusing burst of CSI that does not match what the user is doing on camera.
- **Ctrl+C in the orchestrator terminal**, not in any subprocess — the orchestrator's signal handler is the only thing that cleans up the GPU correctly.

---

## 8. Summary of Recurring Lessons

- **Decouple processes through the file system, not shared memory.** MediaPipe and Sionna cannot coexist in one Python process, and that constraint turned out to be a feature: per-frame `.ply` files are an obvious, debuggable interface.
- **Cap queues; drop, do not buffer.** Live demos must keep up with reality. A bounded queue with frame dropping is far better than an unbounded queue with growing latency.
- **Tag every result with its frame ID.** Out-of-order completion is the default in parallel pipelines; ordering must be reconstructed by the collector, not assumed.
- **Restrict the `.blend` file to a stable, version-controlled artefact.** Anything live happens in memory only — never on disk.
