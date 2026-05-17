# Sionna Environment Setup (`sionna_env/`)

The `sionna_env/` directory at `02_rf_simulation/sionna_env/` holds the Python virtual environment used by every RF simulation script in this module and by the parallel Sionna worker pool in `05_live_demo/`. It is excluded from Git (see `.gitignore`), so after cloning the repository you need to recreate it locally before any RF simulation script will run.

This document records the exact procedure used to build the environment, the package versions known to work, and the common setup problems to avoid.

---

## 1. Prerequisites

- **Python 3.10** specifically. Sionna 1.2.x and TensorFlow 2.20.x are pinned to this version in this project. Newer Python (3.11/3.12/3.13/3.14) ships with ABI changes that the bundled Mitsuba binaries do not support, and older Python lacks features Sionna depends on. Do not use the system Python on macOS — install a clean 3.10 via Homebrew or `pyenv`.
- **macOS or Linux** with at least 8 GB RAM. A CUDA-capable GPU is strongly recommended for the live demo but not required for offline batch simulation.
- **NVIDIA GPU + recent CUDA drivers** if you want GPU acceleration. On macOS, TensorFlow runs CPU-only by default; that is fine for development but slow for full dataset generation.
- Git, `curl`, and a working C/C++ toolchain (Xcode Command Line Tools on macOS, `build-essential` on Linux). Some Sionna dependencies build native extensions on install.

---

## 2. Installing Python 3.10

### macOS (Homebrew)

```bash
brew install python@3.10
which python3.10
# /opt/homebrew/opt/python@3.10/bin/python3.10  (Apple Silicon)
# /usr/local/opt/python@3.10/bin/python3.10     (Intel Mac)
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev build-essential
which python3.10
```

### Cross-platform (pyenv)

```bash
pyenv install 3.10.19
pyenv shell 3.10.19
which python
```

Verify the version before continuing:

```bash
python3.10 --version
# Python 3.10.19   (or a close 3.10.x — anything in 3.10 is fine)
```

---

## 3. Creating the Virtual Environment

From the **repository root**, run:

```bash
cd 02_rf_simulation
python3.10 -m venv sionna_env
```

This creates a fresh, isolated environment at `02_rf_simulation/sionna_env/`. The directory layout you should see:

```
02_rf_simulation/sionna_env/
├── bin/            # python, pip, activate, ...
├── include/
├── lib/python3.10/site-packages/
├── pyvenv.cfg
└── share/
```

**Use `--no-system-site-packages` (the default).** Do not pass `--system-site-packages`. Sionna ships its own Mitsuba binary and will misbehave if a system-wide Mitsuba is on the import path.

---

## 4. Activating the Environment

```bash
source 02_rf_simulation/sionna_env/bin/activate
```

After activation your shell prompt should show `(sionna_env)` and `which python` should resolve inside the venv:

```bash
which python
# .../02_rf_simulation/sionna_env/bin/python
python --version
# Python 3.10.19
```

Deactivate with `deactivate` when you are done.

---

## 5. Installing Packages

With the environment activated, upgrade `pip` first and then install the pinned package set:

```bash
pip install --upgrade pip setuptools wheel

# Core RF simulation stack
pip install sionna==1.2.1
pip install tensorflow==2.20.0
pip install mitsuba==3.7.1
pip install drjit==1.2.0

# Numerical and plotting
pip install numpy==2.2.6
pip install scipy==1.15.3
pip install matplotlib==3.10.8

# Optional but useful for inspection
pip install ipython jupyter
```

The exact versions above are the ones present in the working `sionna_env/` and are known to be mutually compatible. A single-line install is also possible if you trust pip's resolver:

```bash
pip install sionna==1.2.1 tensorflow==2.20.0 mitsuba==3.7.1 drjit==1.2.0 \
            numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.8
```

Installing Sionna pulls in Mitsuba and Dr.Jit as transitive dependencies, but pinning them explicitly avoids the resolver picking a newer Mitsuba that breaks scene loading.

> **Why these versions?** The main repo `requirements.txt` lists looser bounds (`sionna>=0.17`, `tensorflow>=2.13`), which is fine for first-time installation but does not guarantee reproducibility. The versions above are what was actually validated end-to-end against the scripts in this module.

---

## 6. Verifying the Installation

A quick smoke test confirms that Sionna, TensorFlow, and Mitsuba all import cleanly:

```bash
source 02_rf_simulation/sionna_env/bin/activate
python - <<'PY'
import sionna, tensorflow as tf, mitsuba as mi, drjit as dr
print("sionna     :", sionna.__version__)
print("tensorflow :", tf.__version__)
print("mitsuba    :", mi.__version__)
print("drjit      :", dr.__version__)
print("GPUs visible:", tf.config.list_physical_devices("GPU"))
PY
```

Expected output (versions should match section 5; GPU list will be empty on CPU-only machines):

```
sionna     : 1.2.1
tensorflow : 2.20.0
mitsuba    : 3.7.1
drjit      : 1.2.0
GPUs visible: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If all four imports succeed, the environment is ready. To verify the actual simulation path also works, run a single-frame inspection:

```bash
python scripts/load_npz.py data/mmwave_gesture_sample.npy
```

This should print the array shape without errors.

---

## 7. Common Setup Problems

### 7.1 `ModuleNotFoundError: No module named 'sionna'` even after install

You activated the wrong shell. Check `which python` — it must point inside `sionna_env/bin/`. Activations do not persist across new terminal tabs; activate again in each new shell.

### 7.2 `mitsuba` import fails with a symbol-mismatch error

A system-wide `mitsuba` package is shadowing the venv-bundled one. Recreate the venv with `--no-system-site-packages` (the default), and confirm with:

```bash
python -c "import mitsuba; print(mitsuba.__file__)"
```

The path should be inside `02_rf_simulation/sionna_env/lib/python3.10/site-packages/mitsuba/`.

### 7.3 TensorFlow installation pulls in CUDA wheels you did not ask for

On Linux, `pip install tensorflow==2.20.0` includes the CUDA 12 wheel by default. If you want CPU-only (smaller install, no driver requirements):

```bash
pip install tensorflow-cpu==2.20.0
```

instead of `tensorflow`. Sionna runs fine on CPU for offline simulation; only the live demo materially benefits from a GPU.

### 7.4 `pip install sionna` resolves to a newer version than 1.2.1

Pin explicitly with `sionna==1.2.1`. The Sionna API has changed across point releases and the scripts in this module call the v1.2.x PathSolver signature. A newer Sionna will raise `TypeError` from `save_rf_parallel_v2.py`.

### 7.5 Apple Silicon: `tensorflow` install fails with `metal` errors

On Apple Silicon, install the regular `tensorflow` wheel (not `tensorflow-macos` and not `tensorflow-metal`). The combination of TF 2.20 + Sionna 1.2.1 we use does not require the Metal plugin; adding it tends to break Sionna's TF detection.

### 7.6 GPU memory exhausted with multiple workers

If you launch the parallel Sionna worker pool and TensorFlow reports OOM on the second worker, that is a runtime problem, not a setup problem. See `RF_SIMULATION_NOTES.md` §4.1 for the `set_memory_growth` fix that the scripts apply automatically.

---

## 8. Re-creating from Scratch

If anything in the environment goes wrong and you want a clean rebuild, the safest path is:

```bash
deactivate                                  # if currently active
rm -rf 02_rf_simulation/sionna_env
cd 02_rf_simulation
python3.10 -m venv sionna_env
source sionna_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install sionna==1.2.1 tensorflow==2.20.0 mitsuba==3.7.1 drjit==1.2.0 \
            numpy==2.2.6 scipy==1.15.3 matplotlib==3.10.8
```

The full rebuild takes about 5–10 minutes on a typical broadband connection (most of the time is the TensorFlow wheel download).

---

## 9. Where This Environment Is Used

- All scripts in `02_rf_simulation/scripts/` and `02_rf_simulation/exploration_scripts/`.
- The parallel Sionna worker pool in `05_live_demo/live_rf_plot_hand/sionna_live_parallel_rf.py`.
- The live CSI plotter `05_live_demo/live_rf_plot_hand/live_rf_plot.py`.
- The `run_pipeline.py` orchestrator in `05_live_demo/` references this environment by absolute path via the `SIONNA_PYTHON` variable — update that path if you place the venv anywhere other than `02_rf_simulation/sionna_env/`.
