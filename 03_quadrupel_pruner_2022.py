# 03_quadrupel_pruner_2022.py
# =====================================================================
# MODUL 3 (TEIL 1 VON 3): EVOLUTIONÄRER QUAD-PRUNER (MASTER)
# =====================================================================
import os
import numpy as np
import pandas as pd

# Parameter-Steuerung für die Zucht-Zyklen (1 Zyklus = 4 Exmissionen)
CYCLES = 7

file_in_A = "02_WB44-2022_A_Normalized.csv"
file_out_B = "03_WB44-2022_A_7xReduced.csv"
log_file = "quadrupel-pruning_wb.log"

if not os.path.exists(file_in_A):
    raise FileNotFoundError(f"A-Datei '{file_in_A}' fehlt")

# 1. Import der als Master normierten Weltbank-Schnittdaten
df_raw = pd.read_csv(file_in_A, dtype=str)
original_column_order = df_raw.columns.tolist()

# Ermittlung der aktiven Kennzahlen (gdpPC, lifeE, etc.)
features = [
    c
    for c in original_column_order
    if c not in ["ID_Number", "country", "iso3", "year"]
]

# 2. Normalisierte Daten nach ID,Jahr,Land und Kuerzel ordnen
current_df = pd.DataFrame()
for col in original_column_order:
    if col in ["ID_Number", "year"]:
        current_df[col] = df_raw[col].str.strip().astype(int)
    elif col in ["country", "iso3"]:
        current_df[col] = df_raw[col].astype(str)
    else:
        current_df[col] = df_raw[col].str.strip().astype(float)

# Pruning-Protokoll initialisieren
with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"Quadrupel-Pruning START BEI N={len(current_df)} ===\n")

# =====================================================================
# MODUL 3 (TEIL 2 VON 3): Quadrupel-PRUNINGSCHLEIFE
# =====================================================================
for cycle in range(1, CYCLES + 1):
    current_n = len(current_df)
    if current_n <= 4:
        print(f"Abbruch in Zyklus {cycle}: Kritische Populationsgrenze!")
        break

    # Matrix für die temporäre Re-Zentrierung extrahieren
    X_mat = current_df[features].values
    X_mat_scaled = (X_mat - np.mean(X_mat, axis=0)) / np.std(X_mat, axis=0)

    # Kovarianz und Symmetrie-Sortierung der schrumpfenden Generation
    cov_matrix = np.cov(X_mat_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Projektion in aktuellen 2PC-Raum zur Fadenkreuz-Schnitt-Analyse
    p1 = X_mat_scaled @ eigenvectors[:, 0]
    p2 = X_mat_scaled @ eigenvectors[:, 1]

    quadruplet_indices = []

    # Definition der Fadenkreuz-Quadranten (Q1 bis Q4)
    quadrants = [
        (p1 >= 0) & (p2 >= 0),  # Q1 (+,+)
        (p1 < 0) & (p2 >= 0),  # Q2 (-,+)
        (p1 < 0) & (p2 < 0),  # Q3 (-,-)
        (p1 >= 0) & (p2 < 0),  # Q4 (+,-)
    ]

    for mask in quadrants:
        quad_idx = np.where(mask)[0]
        if len(quad_idx) == 0:
            continue

        # Raum-Koordinaten des aktuellen Quadranten 
        coords = np.column_stack((p1[quad_idx], p2[quad_idx]))

        min_mean_dist = float("inf")
        best_candidate_local_idx = 0

        # Suche nach stärkst-redundanten Dichte-Mittelpunkt
        for i in range(len(quad_idx)):
            dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
            mean_dist = np.mean(dists)
            if mean_dist < min_mean_dist:
                min_mean_dist = mean_dist
                best_candidate_local_idx = i

        # Rückübersetzung auf globalen Index der aktuellen Matrix
        quadruplet_indices.append(quad_idx[best_candidate_local_idx])

    # 4 stärkst-redundanten Dichte-Mittelpunkte streichen
    dropped_rows = current_df.iloc[quadruplet_indices]
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Zyklus {cycle:02d} (N={current_n} -> {current_n-4}):\n")
        for _, row in dropped_rows.iterrows():
            f.write(f"  - ISO {row['iso3']} | {row['country']}\n")

    # EXMISSION: Tilgung des Quadrupels aus dem Genpool
    current_df = current_df.drop(
        current_df.index[quadruplet_indices]
    ).reset_index(drop=True)
    print(f" -> Zyklus {cycle:02d} beendet. Rest-Scharm: N={len(current_df)}")

# =====================================================================
# MODUL 3 (TEIL 3 VON 3): GETRIMMTER EXPORT (DATASET B)
# =====================================================================
df_save = pd.DataFrame(index=current_df.index)

# Wiederherstellung des getrimmten Textblock-Formats
for col in original_column_order:
    if col == "ID_Number":
        df_save[col] = current_df[col].apply(lambda x: f"{int(x):05d}")
    elif col in ["country", "iso3"]:
        df_save[col] = current_df[col].astype(str)
    elif col == "year":
        df_save[col] = current_df[col].apply(lambda x: f"{int(x):4d}")
    else:
        # Erhalt des 6 Stellen Nachkommaformats mit Vorzeichen-Padding
        df_save[col] = current_df[col].apply(lambda x: f"{float(x): 5.6f}")

df_save.to_csv(file_out_B, index=False)
print(f"\nquadrupel-pruning in '{file_out_B}'.")
    
    
