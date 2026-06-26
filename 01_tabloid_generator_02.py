# 01_tabloid_generator_02.py
# runs for Normalizer
# runs for QuadPruner
# runs for TwinPlotter
# =====================================================================
# MODUL 1: WELTBANK GENERATOR (ZENTRALE DIFFERENZEN NACH ROLLE)
# =====================================================================
import os
import pandas as pd
import numpy as np

file_raw = "WBANK44_INDIK4.csv"
file_out = "01_Weltbank_Kinetic_Master.csv"

if not os.path.exists(file_raw):
    raise FileNotFoundError(f"Rohdatei '{file_raw}' fehlt im Verzeichnis.")

df_in = pd.read_csv(file_raw, dtype=str)

features_base_raw = ["population", "life_expectancy", "inflation_rate", "NY.GDP.PCAP.CD"]
FEATURE_LABELS = ["popul", "lifeE", "infla", "gdpPC"]

df_calc = pd.DataFrame()
df_calc["country"] = df_in["country"].astype(str).str.strip()
df_calc["iso3"] = df_in["iso3"].astype(str).str.strip()
df_calc["year"] = df_in["year"].str.strip().astype(int)

for fb, fl in zip(features_base_raw, FEATURE_LABELS):
    df_calc[fl] = pd.to_numeric(df_in[fb].str.strip(), errors="coerce")

# NaN-Schildwall für die Basismerkmale
df_calc = df_calc.dropna(subset=FEATURE_LABELS).reset_index(drop=True)
df_calc = df_calc.sort_values(["iso3", "year"]).reset_index(drop=True)

# ZENTRALE DIFFERENZ (Jahr + 1 minus Jahr - 1) / 2 pro Land berechnen
for fl in FEATURE_LABELS:
    # shift(-1) holt das nächste Jahr (t+1), shift(1) das vorherige (t-1)
    next_val = df_calc.groupby("iso3")[fl].shift(-1)
    prev_val = df_calc.groupby("iso3")[fl].shift(1)
    df_calc[f"d_{fl}"] = (next_val - prev_val) / 2.0

# Da Randjahre (erstes/letztes Jahr eines Landes) kein t-1 oder t+1 haben -> droppen
df_calc = df_calc.dropna().reset_index(drop=True)

# Pingelige Reformatierung auf 6 Nachkommastellen und gleiche Block-Breite
def format_strict_block(series):
    max_vorkomma = max(len(str(int(abs(val)))) for val in series)
    formatted = []
    for val in series:
        sign = "-" if val < 0 else " "
        v_padded = f"{int(abs(val))}".zfill(max_vorkomma)
        n_part = f"{abs(val) - int(abs(val)):.6f}"[2:]
        formatted.append(f"{sign}{v_padded}.{n_part}")
    return formatted

# Formatierung auf alle Kennzahlen anwenden (year bleibt int)
ALL_8_NUMERIC = FEATURE_LABELS + [f"d_{fl}" for fl in FEATURE_LABELS]
for v in ALL_8_NUMERIC:
    df_calc[v] = format_strict_block(df_calc[v])

# Fortlaufende ID verankern
df_calc.insert(0, "ID_Number", [f"{i:05d}" for i in range(len(df_calc))])

df_calc.to_csv(file_out, index=False)
print(f"Modul 1 ERFOLG: {len(df_calc)} halbglatte Zeilen in '{file_out}' gespeichert.")
