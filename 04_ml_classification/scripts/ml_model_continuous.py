import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# =============================
# SAVE DIR
# =============================
SAVE_DIR = "results_temporal"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# LOAD DATA
# =============================
data = np.load("linked/linked_all_frames.npz", allow_pickle=True)

H = data["H_mat"]
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
# CLEAN (KEEP TRANSITIONS INITIALLY)
# =============================
valid_mask = np.array([
    g is not None and g != ""
    for g in gestures
])

H = H[valid_mask]
gestures = gestures[valid_mask]

print("Initial samples:", len(gestures))

# =============================
# FRAME FEATURES
# =============================
eps = 1e-8

X_frame = np.concatenate([
    np.real(H),
    np.imag(H)
], axis=1)

X_frame = np.nan_to_num(X_frame)
X_frame = np.clip(X_frame, -3, 3)

X_frame = X_frame / (np.linalg.norm(X_frame, axis=1, keepdims=True) + eps)

# =============================
# TEMPORAL WINDOW
# =============================
window = 20
half = window // 2

X_seq, g_seq = [], []

for i in range(half, len(X_frame) - half):
    seq = X_frame[i-half:i+half+1].reshape(-1)
    X_seq.append(seq)
    g_seq.append(gestures[i])

X_seq = np.array(X_seq)
g_seq = np.array(g_seq)

print("After windowing:", len(X_seq))

# =============================
# REMOVE TRANSITIONS
# =============================
mask = np.array(["transition" not in g.lower() for g in g_seq])

X_seq = X_seq[mask]
g_seq = g_seq[mask]

print("After removing transitions:", len(X_seq))
print("Class distribution:", Counter(g_seq))

# =============================
# 🔥 RE-ENCODE LABELS (IMPORTANT FIX)
# =============================
le = LabelEncoder()
y_seq = le.fit_transform(g_seq)

# =============================
# SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# =============================
# PREPROCESS
# =============================
X_train = np.tanh(X_train * 0.5)
X_test  = np.tanh(X_test * 0.5)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + eps)
X_test  = X_test  / (np.linalg.norm(X_test, axis=1, keepdims=True) + eps)

print("Final feature dim:", X_train.shape[1])

# =============================
# TSNE SAVE FUNCTION
# =============================
def save_tsne(X, y, name):
    max_samples = 1500
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(7,6))

    for label in np.unique(y):
        idx = (y == label)
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            label=le.inverse_transform([label])[0],
            alpha=0.7
        )

    plt.legend()
    plt.title(name)

    plt.savefig(os.path.join(SAVE_DIR, f"{name}_tsne.png"))
    plt.close()

# =============================
# EVALUATION FUNCTION
# =============================
def evaluate_model(name, model):
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    labels = np.unique(y_test)
    class_names = le.inverse_transform(labels)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names
    )
    print(report)

    with open(os.path.join(SAVE_DIR, f"{name}_report.txt"), "w") as f:
        f.write(report)

    # confusion matrix (normalized)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.title(f"{name} Confusion Matrix")
    plt.savefig(os.path.join(SAVE_DIR, f"{name}_confusion.png"))
    plt.close()

    # TRUE labels
    save_tsne(X_test, y_test, f"{name}_tsne_true")

    # PREDICTED labels
    save_tsne(X_test, y_pred, f"{name}_tsne_pred")

# =============================
# MODELS
# =============================
evaluate_model("KNN", KNeighborsClassifier(
    n_neighbors=7,
    metric='cosine',
    weights='distance'
))

evaluate_model("SVM", SVC(
    kernel='rbf',
    C=10,
    gamma='auto'
))

evaluate_model("DecisionTree", DecisionTreeClassifier(
    max_depth=25,
    min_samples_leaf=10,
    min_samples_split=20,
    random_state=42
))

evaluate_model("RandomForest", RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
))

evaluate_model("XGBoost", XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=len(np.unique(y_train)),
    eval_metric='mlogloss',
    tree_method='hist',
    n_jobs=-1,
    random_state=42
))

print("\n All results saved in:", SAVE_DIR)