import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.manifold import TSNE

import xgboost as xgb
from collections import Counter

# =============================
# CREATE OUTPUT DIR
# =============================
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# LOAD DATA
# =============================
# change this path to your actual linked dataset
# data = np.load("/Users/dinesh/Documents/mtp/hand_models/linked/linked_all_frames.npz", allow_pickle=True)
data = np.load("linked/linked_all_frames.npz", allow_pickle=True)

H   = data["H_mat"]
CIR = data["CIR_mat"]
gestures = data["gestures"]

# =============================
# MERGE CLASSES
# =============================
gestures = np.array([
    "OPEN" if g is not None and g.lower() in ["wave_left", "wave_right"]
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

print("Remaining samples:", len(gestures))
print("Class distribution:", Counter(gestures))

# =============================
# LABEL ENCODING
# =============================
le = LabelEncoder()
y = le.fit_transform(gestures)

# =============================
# FEATURE ENGINEERING
# =============================
eps = 1e-8

H_mag  = np.log1p(np.abs(H))
CIR_mag = np.log1p(np.abs(CIR))

X = np.concatenate([H_mag, CIR_mag], axis=1)
X = np.nan_to_num(X)
X = np.clip(X, -10, 10)

row_norms = np.linalg.norm(X, axis=1)
valid_rows = row_norms > 1e-12

X = X[valid_rows]
y = y[valid_rows]

X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

# =============================
# SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# SCALING
# =============================
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# =============================
# REMOVE LOW VAR FEATURES
# =============================
var = np.var(X_train, axis=0)
mask = var > 1e-6

X_train = X_train[:, mask]
X_test  = X_test[:, mask]

print("Final feature dimension:", X_train.shape[1])

# =============================
# MODEL DICTIONARY
# =============================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced'),
    "DecisionTree": DecisionTreeClassifier(max_depth=10),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=len(np.unique(y))
    )
}

# =============================
# EVALUATION FUNCTION
# =============================
def evaluate_model(name, model):
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # Save report
    with open(os.path.join(SAVE_DIR, f"{name}_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(os.path.join(SAVE_DIR, f"{name}_confusion.png"))
    plt.close()

    return y_pred

# =============================
# RUN ALL MODELS
# =============================
all_preds = {}

for name, model in models.items():
    all_preds[name] = evaluate_model(name, model)

# =============================
# TSNE VISUALIZATION
# =============================
print("\nGenerating t-SNE...")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_test)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_embedded[:,0],
    X_embedded[:,1],
    c=y_test,
    cmap='tab10',
    s=20
)

plt.title("t-SNE Visualization (True Labels)")
plt.colorbar(scatter, ticks=range(len(le.classes_)))
plt.savefig(os.path.join(SAVE_DIR, "tsne_true.png"))
plt.close()

# =============================
# OPTIONAL: TSNE WITH PREDICTIONS (BEST MODEL)
# =============================
best_model_name = max(all_preds, key=lambda k: accuracy_score(y_test, all_preds[k]))
print("Best model:", best_model_name)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_embedded[:,0],
    X_embedded[:,1],
    c=all_preds[best_model_name],
    cmap='tab10',
    s=20
)

plt.title(f"t-SNE (Predicted - {best_model_name})")
plt.colorbar(scatter, ticks=range(len(le.classes_)))
plt.savefig(os.path.join(SAVE_DIR, "tsne_pred.png"))
plt.close()

print("\n All results saved in:", SAVE_DIR)