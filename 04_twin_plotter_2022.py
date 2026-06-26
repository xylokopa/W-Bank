# 04_twin_plotter_2022.py
# =====================================================================
# MODUL 4 (TEIL 1 VON 3): MULTI-TYPE PLOTTER-KERN (ATTRIBUTE-FIX)
# =====================================================================
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np
import pandas as pd

monitor_title="WELTBANK\nKORRELATIONS-VERGLEICH 2022\n\nISO-Code eingeben\nfür unskalierten GDP"
file_A = "02_WB44-2022_A_Normalized.csv"
file_B = "03_WB44-2022_VerglAuswahl.csv"
file_C = "WBANK44_INDIK4.csv"

df_orig = pd.read_csv(file_A)
df_redu = pd.read_csv(file_B)

# STRIP-SCHUTZ: Header-Namen von versteckten Leerzeichen befreien
df_orig.columns = df_orig.columns.str.strip()
df_redu.columns = df_redu.columns.str.strip()

# DYNAMISCHE ERKENNUNG: Alles außer Metadaten wird zum Feature
ALL_FEATURES = [
    c for c in df_orig.columns 
    if c.strip() not in ["ID_Number", "country", "iso3", "year"]
]
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
    """Echte dimetrische Reiterperspektive (Pferd statt Frosch)"""
    alpha, beta = np.radians(7.0), np.radians(42.0)
    kx, ky, kz = 0.95, 1.0, 0.5
    x_2d = x_3d * kx * np.cos(alpha) - z_3d * kz * np.cos(beta)
    y_2d = y_3d * ky - x_3d * kx * np.sin(alpha) - z_3d * kz * np.sin(beta)
    return x_2d, y_2d

def project_dimetric(X_scaled):
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

    # CRASH-FIX: Erst erzwungene Text-Wandlung (.astype(str)), dann Whitespace-Kappung
    X_A = df_orig[active_feats].astype(str).stack().str.strip().astype(float).unstack().values
    cov_orig = np.cov(X_A, rowvar=False)
    evals_A, evecs_A = np.linalg.eigh(cov_orig)
    idx_A = np.argsort(evals_A)[::-1]
    evals_A_raw, evecs_A_raw = evals_A[idx_A], evecs_A[:, idx_A]

    X_B = df_redu[active_feats].astype(str).stack().str.strip().astype(float).unstack().values
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

# FIGUR-LAYOUT INITIALISIEREN
fig = plt.figure(figsize=(16, 9.5))
plt.rcParams["toolbar"] = "None"
ax_left = plt.subplot2grid((2, 6), (0, 0), colspan=3)
ax_right = plt.subplot2grid((2, 6), (0, 3), colspan=3)
ax_cov_A = plt.subplot2grid((2, 6), (1, 0), colspan=2)
ax_cov_B = plt.subplot2grid((2, 6), (1, 2), colspan=2)
ax_load = plt.subplot2grid((2, 6), (1, 4), colspan=2)
plt.subplots_adjust(bottom=0.15, hspace=0.35, wspace=0.45)
highlight_orig, highlight_redu = None, None
#######################################################################
# =====================================================================
# MODUL 4 (TEIL 2 VON 3): RENDERING ENGINES & STRING-CLEANED RAW MONITOR
# =====================================================================


def draw_background_axes(ax, active_feats):
    """Baut das räumliche graue Koordinatensystem im Schrägbild"""
    if not is_3d_mode or len(active_feats) != 3:
        return
    axis_len = 2.0
    axes_3d = [
        (axis_len, 0, 0, active_feats, "darkred"),
        (0, axis_len, 0, active_feats, "darkgreen"),
        (0, 0, axis_len, active_feats, "darkblue"),
    ]
    for x3d, y3d, z3d, label, col in axes_3d:
        x2d, y2d = project_dimetric_core(x3d, y3d, z3d)
        ax.plot(
            [0, x2d],
            [0, y2d],
            color="darkgray",
            linestyle=":",
            linewidth=1.2,
            zorder=2,
        )
        ax.text(
            x2d * 1.1,
            y2d * 1.1,
            label,
            color=col,
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            zorder=3,
        )


def draw_eigen_vectors(ax, evecs, evals, is_dataset_b=False):
    """Vektorpfeile als Skalare übergeben - blockiert jeden Inhomogenitäts-Crash"""
    if not is_3d_mode or evecs is None or len(evals) < 2:
        return
    len_pc1 = np.sqrt(evals) * 1.5
    len_pc2 = np.sqrt(evals) * 1.5

    v1_3d = evecs[:, 0] * len_pc1
    v2_3d = evecs[:, 1] * len_pc2

    if is_dataset_b:
        if rotation_state == 1:
            v1_3d, v2_3d = v2_3d, -v1_3d
        elif rotation_state == 2:
            v1_3d, v2_3d = -v1_3d, -v2_3d
        elif rotation_state == 3:
            v1_3d, v2_3d = -v2_3d, v1_3d
        v1_3d = v1_3d * mirror_state

    # ABSOLUTER SKALAR-ABGRIFF
    v1_2d_x, v1_2d_y = project_dimetric_core(v1_3d, v1_3d, v1_3d)
    v2_2d_x, v2_2d_y = project_dimetric_core(v2_3d, v2_3d, v2_3d)

    ax.arrow(
        0, 0, v1_2d_x, v1_2d_y, color="teal", width=0.03, head_width=0.12, zorder=10
    )
    ax.arrow(
        0,
        0,
        v2_2d_x,
        v2_2d_y,
        color="orange",
        width=0.03,
        head_width=0.12,
        zorder=10
    )


def draw_economic_monitor(ax):
    """Holt das BIP fehlersicher aus der Ur-CSV und plottet absolut stabil"""
    ax.cla()
    active_idx = idx_focus_B if idx_focus_B is not None else idx_focus_A
    current_df = df_redu if idx_focus_B is not None else df_orig

    if active_idx is None or current_df.empty:
        ax.text(0.5,0.5,monitor_title,ha="center",va="center",color="gray",
                fontsize=9,weight="bold",)
        ax.set_axis_off()
        return
    try:
        # STRING-FIX: Wir erzwingen die Extraktion des nackten Textes aus der Zeilenserie!
        raw_iso = current_df["iso3"].iloc[active_idx]
        if isinstance(raw_iso, pd.Series):
            raw_iso = raw_iso.iloc[0]
        target_iso = str(raw_iso).strip().upper()

        raw_country = current_df["country"].iloc[active_idx]
        if isinstance(raw_country, pd.Series):
            raw_country = raw_country.iloc[0]
        target_country = str(raw_country).strip()

        # Einlesen der originalen Weltbank-Basisdatei
        df_raw_wb = pd.read_csv(file_C, dtype=str)

        # Spalten-Sensor: Finde die Spalte für das Geld
        gdp_col = None
        for col in df_raw_wb.columns:
            if "GDP" in col.upper() or "PCAP" in col.upper():
                gdp_col = col
                break
        if gdp_col is None:
            gdp_col = "NY.GDP.PCAP.CD"

        # Ländercode-Vergleichsspalte im Speicher säubern
        df_raw_wb["iso3_clean"] = (
            df_raw_wb["iso3"].astype(str).str.strip().str.upper()
        )
        df_land = df_raw_wb[df_raw_wb["iso3_clean"] == target_iso].copy()

        if not df_land.empty:
            # Konvertierung über numerischen Schildwall
            df_land["year_num"] = pd.to_numeric(
                df_land["year"].astype(str).str.strip(), errors="coerce"
            )
            df_land["gdp_num"] = pd.to_numeric(
                df_land[gdp_col].astype(str).str.strip(), errors="coerce"
            )

            # Zeilen ohne gültige Werte eliminieren und chronologisch ordnen
            df_land = df_land.dropna(subset=["year_num", "gdp_num"]).sort_values(
                "year_num"
            )

            if not df_land.empty:
                ax.set_axis_on()
                # Stabiler Linienplot ohne Absturzgefahr
                ax.plot(
                    df_land["year_num"].values,
                    df_land["gdp_num"].values,
                    color="crimson",
                    linewidth=2.0,
                )

                ax.set_title(
                    f"GDP-Trend: {target_country}",
                    fontsize=9,
                    weight="bold",
                    color="darkred",
                )
                ax.set_xlabel("Jahr", fontsize=7)
                ax.set_ylabel("GDP pro Kopf (in USD)", fontsize=7)

                # Währungs-Formatierung exakt nach deiner Vorlage
                ax.get_yaxis().set_major_formatter(
                    plt.FuncFormatter(lambda x, loc: "{:,} $".format(int(x)))
                )
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.tick_params(axis="both", labelsize=7)
                return
        raise ValueError
    except Exception:
        ax.text(
            0.5,
            0.5,
            f"Keine GDP-Urdaten für\n{target_iso} in der CSV gefunden.",
            ha="center",
            va="center",
            color="crimson",
            fontsize=9,
        )
        ax.set_axis_off()


def enforce_button_colors():
    for f in ALL_FEATURES:
        c = "orchid" if feature_states[f] else "lightgray"
        buttons_toggle[f].color = c
        buttons_toggle[f].hovercolor = c
        buttons_toggle[f].ax.set_facecolor(c)
#######################################################################
# =====================================================================
# MODUL 4 (TEIL 3 VON 3): SCREEN REFRESH, WIDGETS & MAIN CALLBACKS
# =====================================================================

def update_plots(full_refresh=True):
    global PC1_redu, PC2_redu, highlight_orig, highlight_redu, scatter_B
    active_feats = [f for f in ALL_FEATURES if feature_states[f]]

    if is_3d_mode:
        if rotation_state == 0: p1, p2 = X3D_redu_base.copy(), Y3D_redu_base.copy()
        elif rotation_state == 1: p1, p2 = Y3D_redu_base.copy(), -X3D_redu_base.copy()
        elif rotation_state == 2: p1, p2 = -X3D_redu_base.copy(), -Y3D_redu_base.copy()
        elif rotation_state == 3: p1, p2 = -Y3D_redu_base.copy(), X3D_redu_base.copy()
        p1 = p1 * mirror_state; x_plot_A, y_plot_A = X3D_orig, Y3D_orig; x_plot_B, y_plot_B = p1, p2
    else:
        if rotation_state == 0: p1, p2 = PC1_base_redu.copy(), PC2_base_redu.copy()
        elif rotation_state == 1: p1, p2 = PC2_base_redu.copy(), -PC1_base_redu.copy()
        elif rotation_state == 2: p1, p2 = -PC1_base_redu.copy(), -PC2_base_redu.copy()
        elif rotation_state == 3: p1, p2 = -PC2_base_redu.copy(), PC1_base_redu.copy()
        p1 = p1 * mirror_state; x_plot_A, y_plot_A = PC1_orig, PC2_orig; x_plot_B, y_plot_B = p1, p2

    PC1_redu, PC2_redu = x_plot_B, y_plot_B

    if full_refresh:
        ax_left.cla(); ax_right.cla()
        for ax in [ax_left, ax_right]:
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Perspektive X" if is_3d_mode else "PC1")
            ax.set_ylabel("Perspektive Y" if is_3d_mode else "PC2")

        if is_3d_mode and len(active_feats) == 3:
            draw_background_axes(ax_left, active_feats); draw_background_axes(ax_right, active_feats)

        ax_left.scatter(x_plot_A, y_plot_A, alpha=0.4, color="blue")
        (highlight_orig,) = ax_left.plot([], [], "ro", markersize=10, zorder=12)
        ax_left.set_title(f"A:Grundmenge")
        for i in range(len(x_plot_A)):
            ax_left.text(x_plot_A[i] + 0.05, y_plot_A[i] + 0.05, str(df_orig["iso3"].iloc[i]), fontsize=7, alpha=0.3)

        scatter_B = ax_right.scatter(x_plot_B, y_plot_B, alpha=0.6, color="royalblue")
        (highlight_redu,) = ax_right.plot([], [], "ro", markersize=10, zorder=12)
        ax_right.set_title(f"B:Vergleichs-Auswahl")
        for i in range(len(x_plot_B)):
            ax_right.text(x_plot_B[i] + 0.05, y_plot_B[i] + 0.05, str(df_redu["iso3"].iloc[i]), fontsize=8, alpha=0.5)

        if is_3d_mode:
            draw_eigen_vectors(ax_left, evecs_A_raw, evals_A_raw, is_dataset_b=False)
            draw_eigen_vectors(ax_right, evecs_B_raw, evals_B_raw, is_dataset_b=True)

        ax_cov_A.cla(); ax_cov_B.cla()
        for ax, cov, name in [(ax_cov_A, cov_orig, "A"), (ax_cov_B, cov_redu, "B")]:
            ax.imshow(cov, cmap="coolwarm", vmin=-1, vmax=1); ax.set_title(f"Kov-Matrix {name}", fontsize=10)
            ax.set_xticks(range(len(active_feats))); ax.set_yticks(range(len(active_feats)))
            ax.set_xticklabels(active_feats, rotation=25, ha="right", fontsize=8); ax.set_yticklabels(active_feats, fontsize=8)
            for i in range(len(active_feats)):
                for j in range(len(active_feats)): ax.text(j, i, f"{cov[i, j]:.2f}", ha="center", va="center", color="black", fontsize=7)
    else:
        if scatter_B is not None:
            scatter_B.set_offsets(np.column_stack((x_plot_B, y_plot_B)))
            if is_3d_mode:
                for p in list(ax_right.patches): p.remove()
                draw_background_axes(ax_right, active_feats)
                draw_eigen_vectors(ax_right, evecs_B_raw, evals_B_raw, is_dataset_b=True)

    if idx_focus_A is not None:
        highlight_orig.set_data([x_plot_A[idx_focus_A]], [y_plot_A[idx_focus_A]])
        ax_left.set_title(f"A: {df_orig['country'].iloc[idx_focus_A]}")
    if idx_focus_B is not None:
        highlight_redu.set_data([x_plot_B[idx_focus_B]], [y_plot_B[idx_focus_B]])
        ax_right.set_title(f"B: {df_redu['country'].iloc[idx_focus_B]}")

    draw_economic_monitor(ax_load)
    enforce_button_colors()
    fig.canvas.draw_idle()

def silence_hover(widget_button):
    widget_button.hovercolor = widget_button.color; widget_button._motion_notify = lambda event: None

# COCKPIT LAYOUT & WIDGET-INITIALISIERUNG
buttons_toggle = {}
x_start = 0.10
for idx, feat in enumerate(ALL_FEATURES):
    ax_btn = plt.axes([x_start + (idx * 0.041), 0.03, 0.039, 0.04])
    btn = Button(ax_btn, feat[:6], color="orchid"); silence_hover(btn); buttons_toggle[feat] = btn

def make_toggle_callback(f_name):
    def callback(event):
        global filter_lock
        if filter_lock: return
        if sum(feature_states.values()) <= 2 and feature_states[f_name]: return
        filter_lock = True; feature_states[f_name] = not feature_states[f_name]
        if recalculate_pca(): update_plots(full_refresh=True)
        filter_lock = False
    return callback

for feat in ALL_FEATURES: buttons_toggle[feat].on_clicked(make_toggle_callback(feat))

ax_btn_restart = plt.axes([0.45, 0.03, 0.05, 0.04])
btn_restart = Button(ax_btn_restart, "restart", color="crimson"); silence_hover(btn_restart)

def on_restart(event):
    global rotation_state, mirror_state, filter_lock, idx_focus_A, idx_focus_B
    if filter_lock: return
    filter_lock = True; rotation_state, mirror_state = 0, 1; idx_focus_A, idx_focus_B = None, None
    for f in ALL_FEATURES: feature_states[f] = True
    recalculate_pca(); update_plots(full_refresh=True); filter_lock = False
btn_restart.on_clicked(on_restart)

ax_box_l = plt.axes([0.535, 0.03, 0.035, 0.04]); text_box_l = TextBox(ax_box_l, "ISO A: ", initial="")
ax_box_r = plt.axes([0.605,0.03, 0.035, 0.04]); text_box_r = TextBox(ax_box_r, "ISO B: ", initial="")

def submit_left(text):
    global idx_focus_A
    try:
        val = str(text).strip().upper(); match = df_orig[df_orig["iso3"] == val]
        idx_focus_A = df_orig.index[df_orig["iso3"] == val].tolist() if not match.empty else None
        update_plots(full_refresh=True)
    except Exception: idx_focus_A = None

def submit_right(text):
    global idx_focus_B
    try:
        val = str(text).strip().upper(); match = df_redu[df_redu["iso3"] == val]
        idx_focus_B = df_redu.index[df_redu["iso3"] == val].tolist() if not match.empty else None
        update_plots(full_refresh=True)
    except Exception: idx_focus_B = None

text_box_l.on_submit(submit_left); text_box_r.on_submit(submit_right)

ax_btn_rot = plt.axes([0.68, 0.03, 0.11, 0.04]); btn_rotate = Button(ax_btn_rot, "Drehen (90°)", color="dodgerblue"); silence_hover(btn_rotate)
ax_btn_mir = plt.axes([0.81, 0.03, 0.10, 0.04]); btn_mirror = Button(ax_btn_mir, "Spiegeln (H-Achse)", color="crimson"); silence_hover(btn_mirror)

def rotate_reduktion(event): global rotation_state; rotation_state = (rotation_state + 1) % 4; update_plots(full_refresh=False)
def mirror_reduktion(event): global mirror_state; mirror_state = -1 if mirror_state == 1 else 1; update_plots(full_refresh=False)
btn_rotate.on_clicked(rotate_reduktion); btn_mirror.on_clicked(mirror_reduktion)

# DER INITIALE ZUENDSCHLUESSEL AM ABSOLUTEN ENDE
update_plots(full_refresh=True)
plt.show()
        


