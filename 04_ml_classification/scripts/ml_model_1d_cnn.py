import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
from collections import Counter

# =============================
# SAVE DIR
# =============================
SAVE_DIR = "results_cnn_attention"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# DEVICE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =============================
# LOAD DATA
# =============================
# change this path to your actual linked dataset
# data = np.load("/Users/dinesh/Documents/mtp/hand_models/linked/linked_all_frames.npz", allow_pickle=True)
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
# CLEAN
# =============================
mask = np.array([g is not None and g != "" for g in gestures])
H = H[mask]
gestures = gestures[mask]

# =============================
# FEATURES (REAL + IMAG)
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
window = 100   # keep large window
half = window // 2

X_seq, g_seq = [], []

for i in range(half, len(X_frame) - half):
    seq = X_frame[i-half:i+half+1]
    X_seq.append(seq)
    g_seq.append(gestures[i])

X_seq = np.array(X_seq)
g_seq = np.array(g_seq)

# =============================
# SEQUENCE NORMALIZATION
# =============================
X_seq = X_seq / (np.linalg.norm(X_seq, axis=(1,2), keepdims=True) + eps)

# =============================
# REMOVE TRANSITIONS
# =============================
mask = np.array(["transition" not in g.lower() for g in g_seq])
X_seq = X_seq[mask]
g_seq = g_seq[mask]

print("Samples:", len(X_seq))
print("Class dist:", Counter(g_seq))

# =============================
# LABEL ENCODING
# =============================
le = LabelEncoder()
y = le.fit_transform(g_seq)

# =============================
# SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=y
)

# =============================
# TORCH DATA
# =============================
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test, dtype=torch.float32).to(device)

y_train = torch.tensor(y_train).to(device)
y_test  = torch.tensor(y_test).to(device)

# =============================
# SOFT CLASS WEIGHTS
# =============================
counts = np.bincount(y_train.cpu().numpy())

weights = 1.0 / (counts + 1e-6)
weights = weights / np.sum(weights) * len(weights)
weights = np.power(weights, 0.5)

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

# =============================
# CNN + ATTENTION MODEL
# =============================
class CNN1DAttention(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # 🔥 ATTENTION
        self.attn = nn.Linear(256, 1)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, time)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # 🔥 Attention pooling
        x_t = x.permute(0, 2, 1)  # (batch, time, channels)

        weights = torch.softmax(self.attn(x_t), dim=1)  # (batch, time, 1)
        x = (x_t * weights).sum(dim=1)  # weighted sum

        x = self.dropout(x)
        x = self.fc(x)

        return x

model = CNN1DAttention(
    input_dim=X_train.shape[2],
    num_classes=len(le.classes_)
).to(device)

# =============================
# TRAINING
# =============================
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)

optimizer = optim.Adam(model.parameters(), lr=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=25
)

epochs = 25
batch_size = 128

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train.size(0))

    total_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]

        xb = X_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# =============================
# EVALUATION
# =============================
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

y_true = y_test.cpu().numpy()

labels = np.unique(y_true)
class_names = le.inverse_transform(labels)

report = classification_report(
    y_true, y_pred,
    labels=labels,
    target_names=class_names,
    zero_division=0
)

print("\n===== CNN + ATTENTION =====")
print(report)

with open(os.path.join(SAVE_DIR, "cnn_attention_report.txt"), "w") as f:
    f.write(report)

# =============================
# CONFUSION MATRIX
# =============================
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt=".2f",
            xticklabels=class_names,
            yticklabels=class_names)

plt.title("CNN Attention Confusion Matrix")
plt.savefig(os.path.join(SAVE_DIR, "cnn_attention_confusion.png"))
plt.close()

# =============================
# TSNE
# =============================
def save_tsne(X, y, name):
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
    plt.savefig(os.path.join(SAVE_DIR, f"{name}.png"))
    plt.close()

X_test_np = X_test.cpu().numpy().reshape(X_test.shape[0], -1)

save_tsne(X_test_np, y_true, "tsne_true")
save_tsne(X_test_np, y_pred, "tsne_pred")

print("\n All results saved in:", SAVE_DIR)