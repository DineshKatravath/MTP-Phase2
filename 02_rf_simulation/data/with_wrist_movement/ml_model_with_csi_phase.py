import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from collections import Counter

# =============================
# SAVE DIRECTORY
# =============================
SAVE_DIR = "results_real_imag"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# LOAD DATA
# =============================
data = np.load("linked/linked_all_frames.npz", allow_pickle=True)

H   = data["H_mat"]
CIR = data["CIR_mat"]
gestures = data["gestures"]

# =============================
# MERGE CLASSES
# =============================
gestures = np.array([
    "OPEN" if g is not None and g.lower() in ["wave_left", "wave_right", "open_palm"]
    else g
    for g in gestures
])

# =============================
# CLEAN DATA
# =============================
valid_mask = np.array([
    g is not None and g != "" and "transition" not in g.lower()
    for g in gestures
])

H = H[valid_mask]
CIR = CIR[valid_mask]
gestures = gestures[valid_mask]

print("Samples:", len(gestures))
print("Class distribution:", Counter(gestures))

# =============================
# LABEL ENCODING
# =============================
le = LabelEncoder()
y = le.fit_transform(gestures)

# =============================
# FEATURE ENGINEERING (REAL + IMAG)
# =============================
eps = 1e-8

X = np.concatenate([
    np.real(H),
    np.imag(H),
    np.real(CIR),
    np.imag(CIR)
], axis=1)

X = np.nan_to_num(X)
X = np.clip(X, -5, 5)

row_norms = np.linalg.norm(X, axis=1)
valid_rows = row_norms > 1e-12

X = X[valid_rows]
y = y[valid_rows]

# normalize per sample
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

# =============================
# TRAIN TEST SPLIT
# =============================
counts = Counter(y)
stratify = y if min(counts.values()) >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# =============================
# PREPROCESSING
# =============================
# squash
X_train = np.tanh(X_train)
X_test  = np.tanh(X_test)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# PCA
pca = PCA(n_components=0.99, whiten=True)
X_train = pca.fit_transform(X_train)
X_test  = pca.transform(X_test)

# normalize again
X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + eps)
X_test  = X_test  / (np.linalg.norm(X_test, axis=1, keepdims=True) + eps)

print("Final feature dim:", X_train.shape[1])

# =============================
# MODELS
# =============================
models = {
    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        metric='cosine',
        weights='distance'
    ),

    "SVM": SVC(
        kernel='rbf',
        C=10,
        gamma='auto'
    ),

    "DecisionTree": DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=10,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=len(np.unique(y_train)),
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
}

# SVM weights (class imbalance)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# =============================
# TRAIN + EVALUATE
# =============================
all_preds = {}
all_acc = {}

for name, model in models.items():
    print(f"\n===== {name} =====")

    if name == "SVM":
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # save report
    with open(os.path.join(SAVE_DIR, f"{name}_report.txt"), "w") as f:
        f.write(report)

    # confusion matrix (normalized)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"{name} Confusion Matrix (Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(os.path.join(SAVE_DIR, f"{name}_confusion.png"))
    plt.close()

    all_preds[name] = y_pred
    all_acc[name] = acc

# =============================
# TSNE VISUALIZATION
# =============================
print("\nGenerating t-SNE...")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_test)

# TRUE LABELS
plt.figure(figsize=(8,6))
plt.scatter(X_embedded[:,0], X_embedded[:,1],
            c=y_test, cmap='tab10', s=20)
plt.title("t-SNE (True Labels)")
plt.savefig(os.path.join(SAVE_DIR, "tsne_true.png"))
plt.close()

# BEST MODEL
best_model = max(all_acc, key=all_acc.get)
print("Best model:", best_model)

plt.figure(figsize=(8,6))
plt.scatter(X_embedded[:,0], X_embedded[:,1],
            c=all_preds[best_model],
            cmap='tab10', s=20)
plt.title(f"t-SNE (Predicted - {best_model})")
plt.savefig(os.path.join(SAVE_DIR, "tsne_pred.png"))
plt.close()

print("\n All outputs saved in:", SAVE_DIR)