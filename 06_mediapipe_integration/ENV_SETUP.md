# MediaPipe Environment Setup (`mediaPipeEnv/`)

The `mediaPipeEnv/` directory at `06_mediapipe_integration/mediaPipe/mediaPipeEnv/` holds the Python virtual environment used by the MediaPipe landmark sender (`sender.py` in this module and in `05_live_demo/live_rf_plot_hand/`). It is excluded from Git (see `.gitignore`), so after cloning the repository you need to recreate it locally before the sender will run.

This document records the exact procedure used to build the environment, the package versions known to work, and the common setup problems to avoid.

---

## 1. Prerequisites

- **Python 3.10 – 3.14.** The working environment was built on Python 3.14.3, but MediaPipe 0.10.x supports the entire 3.10–3.14 range. Pick whichever you already have available; this venv must **not** share Python with the Sionna environment (see §6 below).
- **A webcam.** `sender.py` opens the default camera with OpenCV; any USB or built-in webcam at index 0 works. On laptops with multiple cameras you can override the index via an environment variable.
- **macOS or Linux** with at least 4 GB RAM. MediaPipe HandLandmarker runs comfortably on CPU at 30 fps on commodity hardware — no GPU required.
- Git, `curl`, and a working C/C++ toolchain (Xcode Command Line Tools on macOS, `build-essential` on Linux). OpenCV pulls in some native dependencies on first install.
- **macOS only:** Camera permission for the terminal you launch the sender from. On first run macOS prompts you; if you decline, the camera silently returns black frames and MediaPipe never detects a hand.

---

## 2. Installing Python

### macOS (Homebrew)

```bash
brew install python@3.14         # or python@3.12, @3.11, @3.10 — any work
which python3.14
# /opt/homebrew/opt/python@3.14/bin/python3.14   (Apple Silicon)
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev build-essential
which python3.12
```

(Adjust the version suffix to whichever 3.10–3.14 is available in your distro.)

### Cross-platform (pyenv)

```bash
pyenv install 3.14.3
pyenv shell 3.14.3
which python
```

Verify before continuing:

```bash
python3.14 --version
# Python 3.14.3   (or whichever 3.10–3.14 you picked)
```

The rest of this document uses `python3.14`; substitute your version everywhere.

---

## 3. Creating the Virtual Environment

The MediaPipe model file and the venv live together under `06_mediapipe_integration/mediaPipe/`. From the **repository root** run:

```bash
cd 06_mediapipe_integration/mediaPipe
python3.14 -m venv mediaPipeEnv
```

This creates a fresh, isolated environment at `06_mediapipe_integration/mediaPipe/mediaPipeEnv/`. The directory layout you should see:

```
06_mediapipe_integration/mediaPipe/
├── hand_landmarker.task         # MediaPipe model file (already in Git)
├── reloc.pdf
├── README.md
└── mediaPipeEnv/                # newly created venv
    ├── bin/                     # python, pip, activate, ...
    ├── include/
    ├── lib/python3.14/site-packages/
    └── pyvenv.cfg
```

**Use `--no-system-site-packages` (the default).** Do not pass `--system-site-packages`. MediaPipe pins specific `protobuf` and `numpy` versions, and any system-wide install of either will shadow the venv copy with painful runtime errors.

---

## 4. Activating the Environment

```bash
source 06_mediapipe_integration/mediaPipe/mediaPipeEnv/bin/activate
```

After activation your shell prompt should show `(mediaPipeEnv)` and `which python` should resolve inside the venv:

```bash
which python
# .../06_mediapipe_integration/mediaPipe/mediaPipeEnv/bin/python
python --version
# Python 3.14.3
```

Deactivate with `deactivate` when you are done.

---

## 5. Installing Packages

With the environment activated, upgrade `pip` first and then install the pinned package set:

```bash
pip install --upgrade pip setuptools wheel

# MediaPipe and its core runtime
pip install mediapipe==0.10.33
pip install protobuf==3.20.3
pip install absl-py==2.4.0

# Camera I/O
pip install opencv-python==4.13.0.92
# (or opencv-contrib-python==4.13.0.92 if you need the extras modules)

# Numerical and plotting (matplotlib is optional, used only for debug overlays)
pip install numpy==2.4.3
pip install matplotlib==3.10.8
```

The exact versions above are the ones present in the working `mediaPipeEnv/` and are known to be mutually compatible. A single-line install is also possible:

```bash
pip install mediapipe==0.10.33 protobuf==3.20.3 absl-py==2.4.0 \
            opencv-python==4.13.0.92 numpy==2.4.3 matplotlib==3.10.8
```

> **Why pin `protobuf==3.20.3`?** MediaPipe's generated protobuf bindings target the 3.x line. A newer 4.x or 5.x protobuf will appear to install fine but raise `TypeError` deep inside MediaPipe at first inference. Pin it explicitly.

> **Why `opencv-python` and not `opencv-python-headless`?** The sender uses OpenCV's `cv2.imshow` for the optional preview window. If you do not want a GUI dependency, switch to `opencv-python-headless` and remove any preview windows from `sender.py`.

---

## 6. Why This Has To Be a Separate Environment

It is tempting to combine MediaPipe and Sionna in one venv to avoid maintaining two environments. **Do not do this.**

- Sionna depends on TensorFlow 2.20.x and a specific `protobuf` 4.x.
- MediaPipe depends on `protobuf` 3.20.x.
- Loading both in the same process produces immediate segfaults on macOS and Linux (the protobuf descriptor pool collides).

The pipeline is designed around this constraint: the MediaPipe sender and the Sionna workers run in different OS processes with different Python interpreters, and they communicate over a TCP socket (sender → Blender) and the file system (Blender `.ply` → Sionna workers). Keep the environments separate.

---

## 7. The MediaPipe Model File

`sender.py` loads the HandLandmarker model from `06_mediapipe_integration/mediaPipe/hand_landmarker.task`. This file (~8 MB) **is** committed to the repository, so you do not need to download it separately. If for any reason it goes missing, fetch the latest from the official source:

```bash
cd 06_mediapipe_integration/mediaPipe
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Verify the file size is plausible (around 7–9 MB) before running the sender.

---

## 8. Verifying the Installation

A quick smoke test confirms that MediaPipe and OpenCV both import cleanly and that the camera is accessible:

```bash
source 06_mediapipe_integration/mediaPipe/mediaPipeEnv/bin/activate
python - <<'PY'
import cv2, mediapipe as mp, numpy as np
print("opencv   :", cv2.__version__)
print("mediapipe:", mp.__version__)
print("numpy    :", np.__version__)

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
cap.release()
print("camera OK:", ok, "frame shape:", None if not ok else frame.shape)
PY
```

Expected output (versions should match section 5; the frame shape should be something like `(480, 640, 3)`):

```
opencv   : 4.13.0.92
mediapipe: 0.10.33
numpy    : 2.4.3
camera OK: True frame shape: (480, 640, 3)
```

If `camera OK: False`, the issue is camera permission or a wrong device index — see §10.1 below.

To test the full sender → Blender flow:

1. Activate the venv and start `python sender.py` (it will fail to connect because no receiver is running, but you can confirm webcam + MediaPipe init succeeded from the logs).
2. Open the Blender hand rig, run `blender_receivers/reciver_v4.py` from the Scripting workspace, then re-run the sender.

---

## 9. Re-creating from Scratch

If anything in the environment goes wrong and you want a clean rebuild:

```bash
deactivate                                              # if currently active
rm -rf 06_mediapipe_integration/mediaPipe/mediaPipeEnv
cd 06_mediapipe_integration/mediaPipe
python3.14 -m venv mediaPipeEnv
source mediaPipeEnv/bin/activate
pip install --upgrade pip setuptools wheel
pip install mediapipe==0.10.33 protobuf==3.20.3 absl-py==2.4.0 \
            opencv-python==4.13.0.92 numpy==2.4.3 matplotlib==3.10.8
```

The full rebuild takes about 2–4 minutes.

---

## 10. Common Setup Problems

### 10.1 macOS: camera permission denied (black frames, no hand detection)

The first time you run the sender from a particular terminal app (Terminal.app, iTerm2, VS Code, etc.), macOS prompts for camera access. If you missed or declined the prompt:

1. System Settings → Privacy & Security → Camera.
2. Find your terminal app in the list and toggle it on.
3. Restart the terminal app (the permission is read at process start).

Verify by re-running the smoke test in §8.

### 10.2 Wrong webcam selected

`cv2.VideoCapture(0)` picks whatever the OS reports as the first camera, which may not be the one you want on laptops with both internal and external cameras. Override via environment variable:

```bash
CAMERA_INDEX=1 python sender.py
```

Or pass `--camera 1` if `sender.py` exposes the flag in its argument parser.

### 10.3 `protobuf` version conflict at runtime

If MediaPipe import succeeds but inference raises `TypeError: Couldn't build proto file` or similar, you have a newer protobuf than MediaPipe expects. Fix:

```bash
pip install --force-reinstall protobuf==3.20.3
```

### 10.4 `numpy` ABI mismatch

If `import cv2` or `import mediapipe` warns about a NumPy ABI mismatch, you installed OpenCV or MediaPipe against a different NumPy version than the one currently in the venv. Easiest fix is to reinstall both in the right order:

```bash
pip install --force-reinstall numpy==2.4.3
pip install --force-reinstall opencv-python==4.13.0.92 mediapipe==0.10.33
```

### 10.5 Linux: `libGL.so.1: cannot open shared object file`

OpenCV's GUI windows need the system OpenGL library. On a minimal Linux install:

```bash
sudo apt install libgl1 libglib2.0-0
```

If you do not need any preview windows, switch to `opencv-python-headless` (see §5).

### 10.6 `mediaPipeEnv` works locally but the live demo can't find it

`05_live_demo/live_rf_plot_hand/run_pipeline.py` references the MediaPipe interpreter by absolute path via `SENDER_PYTHON`. Update that path to point at `06_mediapipe_integration/mediaPipe/mediaPipeEnv/bin/python` on your machine.

---

## 11. Where This Environment Is Used

- `06_mediapipe_integration/sender.py` — the standalone MediaPipe sender.
- `05_live_demo/live_rf_plot_hand/sender.py` — the live-demo MediaPipe sender, launched by `run_pipeline.py`.
- Any ad-hoc script that needs to read a webcam and produce hand landmarks.

The Blender receivers (`06_mediapipe_integration/blender_receivers/*.py` and `05_live_demo/live_rf_plot_hand/blender_receiver/reciever_live_frames.py`) do **not** use this environment — they run inside Blender's embedded Python interpreter and only need the standard library plus `bpy`.
