# WB_TwinPlotter_01.py 21-06-2026
# =====================================================================
# MODUL 4: MODULES TWIN-PLOTTER INTERFACE (PART A)
# =====================================================================
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np
import pandas as pd

file_A = "02_Dataset_A_Normalized.csv"
file_B = "03_Dataset_B_Reduced.csv"

df_orig = pd.read_csv(file_A)
df_redu = pd.read_csv(file_B)

# Dynamische Ermittlung der im Schnitt aktiven Variablen
ALL_FEATURES = [c for c in df_orig.columns if c not in ["ID_Number", "country", "iso3"]]
feature_states = {feat: True for feat in ALL_FEATURES}

rotation_state = 0
mirror_state = 1
idx_focus_A, idx_focus_B = None, None
filter_lock = False

PC1_orig, PC2_orig = None, None
PC1_base_redu, PC2_base_redu = None, None
cov_orig, cov_redu = None, None
evecs_redu, evals_redu = None, None
PC1_redu, PC2_redu = None, None
scatter_B = None

is_3d_mode = False
X3D_orig, Y3D_orig = None, None
X3D_redu_base, Y3D_redu_base = None, None
evecs_A_raw, evals_A_raw = None, None
evecs_B_raw, evals_B_raw = None, None

def project_dimetric_core(x_3d, y_3d, z_3d):
    alpha, beta = np.radians(7.0), np.radians(42.0)
    kx, ky, kz = 0.95, 1.0, 0.5
    x_2d = x_3d * kx * np.cos(alpha) - z_3d * kz * np.cos(beta)
    y_2d = y_3d * ky - x_3d * kx * np.sin(alpha) - z_3d * kz * np.sin(beta)
    return x_2d, y_2d

def project_dimetric(X_scaled):
    # Falls das Tripel weniger als 3 Dimensionen hat, absichern
    z = X_scaled[:, 2] if X_scaled.shape[1] > 2 else np.zeros(X_scaled.shape[0])
    return project_dimetric_core(X_scaled[:, 0], X_scaled[:, 1], z)

def recalculate_pca():
    global PC1_orig, PC2_orig, PC1_base_redu, PC2_base_redu, is_3d_mode
    global cov_orig, cov_redu, evecs_redu, evals_redu
    global X3D_orig, Y3D_orig, X3D_redu_base, Y3D_redu_base
    global evecs_A_raw, evals_A_raw, evecs_B_raw, evals_B_raw

    active_feats = [f for f in ALL_FEATURES if feature_states[f]]
    if len(active_feats) < 2: return False

    is_3d_mode = len(active_feats) == 3

    # PCA A
    X_A = df_orig[active_feats].values
    cov_orig = np.cov(X_A, rowvar=False)
    evals_A, evecs_A = np.linalg.eigh(cov_orig)
    idx_A = np.argsort(evals_A)[::-1]
    evals_A_raw, evecs_A_raw = evals_A[idx_A], evecs_A[:, idx_A]

    # PCA B
    X_B = df_redu[active_feats].values
    cov_redu = np.cov(X_B, rowvar=False)
    evals_B, evecs_B = np.linalg.eigh(cov_redu)
    idx_B = np.argsort(evals_B)[::-1]
    evals_redu, evecs_redu = evals_B[idx_B], evecs_B[:, idx_B]
    evals_B_raw, evecs_B_raw = evals_B[idx_B], evecs_B[:, idx_B]

    if is_3d_mode:
        X3D_orig, Y3D_orig = project_dimetric(X_A)
        X3D_redu_base, Y3D_redu_base = project_dimetric(X_B)
    else:
        PC1_orig = X_A @ evecs_A_raw[:, 0]
        PC2_orig = X_A @ evecs_A_raw[:, 1]
        PC1_base_redu = X_B @ evecs_redu[:, 0]
        PC2_base_redu = X_B @ evecs_redu[:, 1]
    return True

recalculate_pca()
PC1_redu = X3D_redu_base.copy() if is_3d_mode else PC1_base_redu.copy()
PC2_redu = Y3D_redu_base.copy() if is_3d_mode else PC2_base_redu.copy()

fig = plt.figure(figsize=(15, 9.5))
plt.rcParams["toolbar"] = "None"
ax_left = plt.subplot2grid((2, 6), (0, 0), colspan=3)
ax_right = plt.subplot2grid((2, 6), (0, 3), colspan=3)
ax_cov_A = plt.subplot2grid((2, 6), (1, 0), colspan=2)
ax_cov_B = plt.subplot2grid((2, 6), (1, 2), colspan=2)
ax_load = plt.subplot2grid((2, 6), (1, 4), colspan=2)
plt.subplots_adjust(bottom=0.15, hspace=0.35, wspace=0.45)
highlight_orig, highlight_redu = None, None
