import os
import numpy as np
import trimesh
from config import *

def get_latest_frame():
    files = sorted([f for f in os.listdir(LIVE_DIR) if f.endswith(".ply")])
    if not files:
        return None

    latest = files[-1]

    # delete backlog
    for f in files[:-1]:
        try:
            os.remove(os.path.join(LIVE_DIR, f))
        except:
            pass

    return latest


def load_mesh(path):
    return trimesh.load(path, process=False)


def compute_scale_factor(mesh):
    verts = np.array(mesh.vertices)
    span = verts.max() - verts.min()
    return TARGET_SPAN_M / span


def normalize_mesh(mesh, sf):
    verts = np.array(mesh.vertices)
    mesh.vertices = verts * sf

    scaled = verts * sf
    center = scaled.mean(axis=0)
    span = scaled.max() - scaled.min()

    return center, span


def calibrate_tx_rx(mesh):
    verts = np.array(mesh.vertices)
    center = verts.mean(axis=0)
    span = verts.max() - verts.min()

    clr = span * CLEARANCE_RATIO

    tx = center - np.array([clr, 0, 0])
    rx = center + np.array([clr, 0, 0])

    return tx, rx


def save_rf(H, CIR, idx):
    path = f"{RF_DIR}/rf_frame_{idx:04d}.npz"
    np.savez(path,
        H_real=np.real(H).astype(np.float32),
        H_imag=np.imag(H).astype(np.float32),
        CIR_real=np.real(CIR).astype(np.float32),
        CIR_imag=np.imag(CIR).astype(np.float32),
    )
    return path