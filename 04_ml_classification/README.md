# 04 — ML Classification

This module evaluates whether the synthetic RF data carries sufficient discriminative information to classify hand gestures. Five classical machine learning classifiers and two CNN variants are trained and evaluated across three feature representations and three dataset variants of increasing difficulty.

## Background

The end goal of the pipeline is to show that distinct hand poses produce well-separated CSI signatures in simulation, and that a machine learning model can reliably recover the hand pose from CSI alone — with no camera at inference time. This module is where that claim is tested quantitatively.

## Gesture Classes

The ML experiments use 8 canonical gesture classes:

| Class | Description |
|-------|-------------|
| `FIST` | All fingers fully curled |
| `OPEN` | Fully extended open palm (includes merged WAVE_LEFT / WAVE_RIGHT) |
| `PINCH` | Thumb and index finger touching |
| `POINT` | Index finger extended, others curled |
| `ROCK` | Index and pinky extended (horns gesture) |
| `THUMBS_DOWN` | Thumb pointing downward |
| `THUMBS_UP` | Thumb pointing upward |
| `V_SIGN` | Index and middle fingers extended |

`WAVE_LEFT` and `WAVE_RIGHT` from the animation library are merged into `OPEN` before training — both have identical finger configurations to an open palm and differ only in wrist orientation; treating them as separate classes would introduce systematic confusion with `OPEN`.

## Dataset Variants

Three labelled datasets were constructed to evaluate classifiers under progressively harder conditions:

| Case | Description | Best Accuracy |
|------|-------------|---------------|
| **Case 1** | Static poses, fixed hand position in room | **100%** (XGBoost, magnitude) |
| **Case 2** | Static poses, hand translates to random positions within room | **86%** (XGBoost, temporal window) |
| **Case 3** | Static poses with rapid random wrist rotation | **59%** (XGBoost, temporal window) |

Each dataset is a linked `.npz` archive from `03_hand_mesh_pipeline/` with arrays `H_mat` (CSI), `CIR_mat`, and `gestures` (string labels). Transition frames are filtered out before training.

## Feature Representations

| Feature | Description | Dimension |
|---------|-------------|-----------|
| **Magnitude** | `|H(f)|` across 256 subcarriers | 256 |
| **Real + Imaginary** | Concatenated real and imaginary parts | 512 |
| **Temporal window** | Stack of T consecutive frames → captures CSI trajectory | 256 × T |

The temporal window representation provided the largest gains in Cases 2 and 3, where the trajectory of CSI change as the hand moves is more class-consistent than any single instantaneous observation.

## Classifiers

| Classifier | Key hyperparameters |
|-----------|---------------------|
| K-Nearest Neighbours (KNN) | k=5, Euclidean distance |
| Support Vector Machine (SVM) | RBF kernel, `RobustScaler` preprocessing |
| Decision Tree | Gini impurity |
| Random Forest | 100 estimators |
| XGBoost | Default gradient boosting parameters |

All classifiers use `RobustScaler` for feature normalisation and scikit-learn's `train_test_split` (80/20 split, stratified by class).

## Scripts

### `scripts/ml_model.py`
Main classification script for Cases 1–3. Loads the linked `.npz` archive, applies feature extraction, trains all 5 classifiers across all 3 feature representations, and saves confusion matrices, classification reports, and t-SNE projections.

### `scripts/ml_model_with_csi_phase.py`
Variant that uses the full complex CSI (magnitude + phase) to assess whether phase information provides additional discriminative power over magnitude alone.

### `scripts/ml_model_1d_cnn.py`
1D CNN classifier operating directly on the raw CSI magnitude vector. Architecture: two Conv1D layers with batch normalisation and max pooling, followed by a dense classification head.

### `scripts/ml_model_continuous.py`
Continuous inference variant for evaluating classifiers on a streaming sequence of frames rather than an IID test split, matching the live demo operating mode.

### `scripts/spectrogram_generator.py`
Generates CSI spectrograms (time vs. subcarrier magnitude) from the linked archive for visual inspection and as input features for the CNN+attention model.

## Results Summary

### Case 1 — Fixed Position

| Classifier | Magnitude | Real+Imag | Temporal |
|-----------|-----------|-----------|---------|
| KNN | ~99% | ~99% | ~99% |
| SVM | ~99% | ~99% | ~99% |
| Decision Tree | ~98% | ~97% | ~98% |
| Random Forest | ~99% | ~99% | ~99% |
| **XGBoost** | **100%** | ~99% | ~99% |

t-SNE projections show 8 compact, non-overlapping clusters — the RF channel acts as a near-perfect pose embedding under fixed-position conditions.

### Case 2 — Position Variation

| Classifier | Best Accuracy | Best Feature |
|-----------|--------------|-------------|
| KNN | ~78% | Temporal |
| SVM | ~75% | Temporal |
| Decision Tree | ~71% | Temporal |
| Random Forest | ~82% | Temporal |
| **XGBoost** | **86%** | **Temporal** |

The 12 percentage point gain of temporal window features over magnitude-only confirms that CSI trajectory over time carries more class-consistent information than any single frame.

### Case 3 — Wrist Rotation

| Classifier | Best Accuracy | Notes |
|-----------|--------------|-------|
| KNN | ~52% | Heavy confusion toward OPEN |
| SVM | ~51% | |
| Decision Tree | ~48% | |
| Random Forest | ~57% | |
| **XGBoost** | **59%** | Above 12.5% random baseline; wrist orientation is primary confound |

Rapid wrist rotation substantially disrupts the CSI latent space structure. Wrist orientation is identified as the dominant factor limiting classification accuracy.

## Directory Structure

```
04_ml_classification/
├── scripts/
│   ├── ml_model.py                  # Main 5-classifier evaluation (Cases 1–3)
│   ├── ml_model_with_csi_phase.py   # Phase-aware CSI classifier
│   ├── ml_model_1d_cnn.py           # 1D CNN classifier
│   ├── ml_model_continuous.py       # Continuous / streaming inference
│   └── spectrogram_generator.py     # CSI spectrogram generation
└── results/
    ├── case1/results/               # Case 1 confusion matrices, reports, t-SNE
    ├── real_imag/results_real_imag/ # Real+imaginary feature results
    ├── temporal/results_temporal/   # Temporal window feature results
    ├── cnn/results_cnn/             # 1D CNN results
    ├── cnn_attention/results_cnn_attention/  # CNN+attention results
    └── spectro_frames*.npz          # Saved spectrogram archives
```

Each results subdirectory contains:
- `<Classifier>_confusion.png` — confusion matrix heatmap
- `<Classifier>_report.txt` — precision, recall, F1 per class
- `tsne_true.png` / `tsne_pred.png` — t-SNE projections coloured by true / predicted label

## Running Classification

```bash
# Activate the Sionna/ML environment
source 02_rf_simulation/sionna_env/bin/activate

# Run the main evaluation (edit the .npz path inside the script)
python scripts/ml_model.py
```

The input `.npz` path is set to `linked/linked_all_frames.npz` by default — update this to match the output of `03_hand_mesh_pipeline/link_rf_to_blender.py`.

## Dependencies

- scikit-learn
- XGBoost
- NumPy, Matplotlib, Seaborn
- TensorFlow / Keras (for CNN variants)

```bash
pip install scikit-learn xgboost matplotlib seaborn tensorflow
```
