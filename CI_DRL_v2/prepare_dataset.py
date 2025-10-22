# prepare_dataset.py
# Read raw_health.csv -> produce lab_health.csv with numeric columns + symptom dummies

import pandas as pd
import numpy as np

RAW = "raw_health.csv"
OUT = "lab_health.csv"

# List of all symptoms discovered earlier (use these exact labels)
SYMPTOMS = [
    "Cough",
    "Fever",
    "Fatigue",
    "Shortness of breath",
    "Runny nose",
    "Headache",
    "Body ache",
    "Sore throat"
]

df = pd.read_csv(RAW)

# Ensure symptom columns exist
symptom_cols = [c for c in df.columns if "Symptom" in c]
if not symptom_cols:
    raise RuntimeError("No Symptom_ columns found in raw CSV")

# split BP into systolic/diastolic
def split_bp(val):
    try:
        s, d = str(val).split("/")
        return float(s), float(d)
    except Exception:
        return (np.nan, np.nan)

syst = []
diast = []
for v in df["Blood_Pressure_mmHg"].fillna("").tolist():
    s, d = split_bp(v)
    syst.append(s); diast.append(d)
df["Systolic_BP"] = syst
df["Diastolic_BP"] = diast
df = df.drop(columns=["Blood_Pressure_mmHg"])

# encode gender
df["Gender"] = df["Gender"].map({"Male": 1, "male":1, "M":1, "Female": 0, "female":0, "F":0}).fillna(0).astype(int)

# Create symptom indicator columns (binary). If any Symptom_i equals one of the known symptoms -> set 1
for s in SYMPTOMS:
    df[f"Symptom_{s}"] = 0

for idx, row in df.iterrows():
    present = set()
    for c in symptom_cols:
        v = row.get(c)
        if pd.isna(v): continue
        v = str(v).strip()
        if v:
            present.add(v)
    for s in SYMPTOMS:
        if s in present:
            df.at[idx, f"Symptom_{s}"] = 1

# Keep only columns we need: Age, Gender, Heart_Rate_bpm, Body_Temperature_C,
# Oxygen_Saturation_%, Systolic_BP, Diastolic_BP, Symptom_* and Diagnosis -> rename to Disease
keep = [
    "Age", "Gender", "Heart_Rate_bpm", "Body_Temperature_C",
    "Oxygen_Saturation_%", "Systolic_BP", "Diastolic_BP"
]
sym_cols = [f"Symptom_{s}" for s in SYMPTOMS]
for c in keep:
    if c not in df.columns:
        raise RuntimeError(f"Required column missing: {c}")

out_df = df[keep + sym_cols + ["Diagnosis"]].copy()
out_df = out_df.rename(columns={"Diagnosis": "Disease"})

# Drop rows with missing essential sensor data
out_df = out_df.dropna(subset=["Age","Heart_Rate_bpm","Body_Temperature_C","Oxygen_Saturation_%","Systolic_BP","Diastolic_BP"]).reset_index(drop=True)

out_df.to_csv(OUT, index=False)
print("Saved cleaned dataset to", OUT)
print("Features saved:", list(out_df.columns))
