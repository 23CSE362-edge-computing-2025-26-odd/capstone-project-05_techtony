# custom_predict.py
# Takes custom input (interactive) and predicts disease

import joblib
import numpy as np
import torch
import torch.nn as nn

# ----- Config -----
SCALER_PATH = "scaler_pg.pkl"
ENCODER_PATH = "labelenc_pg.pkl"
MODEL_PATH = "pg_drl_model_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Load artifacts -----
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)

# same ActorNet as training
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

n_features = len(scaler.mean_)
n_classes = len(le.classes_)
model = ActorNet(n_features, n_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ----- Input Guidelines -----
guidelines = {
    "Heart_Rate_bpm": "60–100 normal, >100 high (tachycardia), <60 low",
    "Body_Temperature_C": "36.1–37.2 normal, >37.5 fever",
    "Systolic_BP": "90–120 normal, >140 high, <90 low",
    "Diastolic_BP": "60–80 normal, >90 high, <60 low",
    "Oxygen_Saturation_%": "95–100 normal, 90–94 mild hypoxia, <90 severe"
}
print("\n=== Input Guidelines ===")
for k,v in guidelines.items(): print(f"{k}: {v}")

# ----- Collect Custom Input -----
try:
    age = float(input("\nEnter Age: "))
    gender = input("Enter Gender (Male/Female): ").strip().lower()
    gender_val = 1 if gender in ["male","m"] else 0

    hr = float(input("Enter Heart Rate (bpm): "))
    temp = float(input("Enter Body Temperature (°C): "))
    spo2 = float(input("Enter Oxygen Saturation (%): "))

    bp = input("Enter Blood Pressure (e.g. 120/80): ")
    try:
        sbp, dbp = bp.split("/")
        sbp, dbp = float(sbp), float(dbp)
    except:
        raise ValueError("Blood Pressure must be systolic/diastolic, e.g. 120/80")

    # feature order must match training dataset
    x = np.array([[age, gender_val, hr, temp, spo2, sbp, dbp]], dtype=np.float32)
    x_scaled = scaler.transform(x)

    with torch.no_grad():
        inp = torch.FloatTensor(x_scaled).to(DEVICE)
        logits = model(inp)
        pred_idx = logits.argmax(dim=1).item()
        pred_class = le.inverse_transform([pred_idx])[0]

    print("\n=== Prediction ===")
    print(f"Predicted Disease: {pred_class}")

except Exception as e:
    print("Error:", e)
