# 02_tomography_normalizer_01.py
# runs from Tabloid 
# runs for  QuadPruner
# runs for  TwinPlotter
# =====================================================================
# MODUL 2: BIVALENTER TOMOGRAPHIE-NORMALIZER (DATASET A GENERATOR)
# =====================================================================
import pandas as pd
import numpy as np

file_master = "01_Weltbank_Kinetic_Master.csv"
file_out_A = "02_Dataset_A_Normalized.csv"

df_master = pd.read_csv(file_master)
# Datentypen zurückgewinnen
for col in df_master.columns:
    if col not in ["ID_Number", "country", "iso3"]:
        df_master[col] = df_master[col].astype(float)

# --- DIE INTERAKTIVE SCHNITTEBENE (Hier wird der Tomograph eingestellt) ---
# Optionen: "QUER" (festes Jahr t), "LÄNGS" (nur d_Features über dt), "SCHRÄG" (Mittelwerte vs Sekanten)
SCHNITT_TYP = "LÄNGS" 
TARGET_YEAR = 2000
DT = 5

features_base = ["popul", "lifeE", "infla", "gdpPC"]
features_deriv = ["d_popul", "d_lifeE", "d_infla", "d_gdpPC"]

if SCHNITT_TYP == "QUER":
    df_slice = df_master[df_master["year"] == TARGET_YEAR].copy()
    selected_vars = features_base + features_deriv
elif SCHNITT_TYP == "LÄNGS":
    df_slice = df_master[(df_master["year"] >= TARGET_YEAR) & (df_master["year"] <= TARGET_YEAR + DT)].copy()
    selected_vars = features_deriv
elif SCHNITT_TYP == "SCHRÄG":
    df_sub = df_master[(df_master["year"] >= TARGET_YEAR) & (df_master["year"] <= TARGET_YEAR + DT)]
    df_slice = df_sub.groupby(["country", "iso3"]).mean(numeric_only=True).reset_index()
    # Ausgewähltes Misch-Tripel aus dem Stirnlappen
    selected_vars = ["gdpPC", "lifeE", "d_gdpPC"]

# Kompromisslose Standardisierung (Z-Transformation)
X = df_slice[selected_vars].values
X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

df_normalized = pd.DataFrame(X_scaled, columns=selected_vars)
df_normalized.insert(0, "iso3", df_slice["iso3"])
df_normalized.insert(0, "country", df_slice["country"])
df_normalized.insert(0, "ID_Number", [f"{i:05d}" for i in range(len(df_slice))])

df_normalized.to_csv(file_out_A, index=False)
print(f"Modul 2 ERFOLG: Schnitt '{SCHNITT_TYP}' normiert unter '{file_out_A}' ({len(df_normalized)} Länder)")
