# predict_feedback.py
# Predict + doctor feedback reinforcement update

import joblib, numpy as np, torch
import torch.nn as nn
import torch.optim as optim

SCALER_PATH = "scaler_pg.pkl"
ENCODER_PATH = "labelenc_pg.pkl"
BEST_MODEL = "pg_drl_model_best.pth"

SYMPTOMS = [
    "Cough","Fever","Fatigue","Shortness of breath",
    "Runny nose","Headache","Body ache","Sore throat"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load artifacts ===
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)

# Model definition
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

# Build feature order
def get_feature_names():
    base = ["Age","Gender","Heart_Rate_bpm","Body_Temperature_C","Oxygen_Saturation_%","Systolic_BP","Diastolic_BP"]
    sym_cols = [f"Symptom_{s}" for s in SYMPTOMS]
    return base + sym_cols

feature_names = get_feature_names()

# === Load model ===
model = ActorNet(len(feature_names), len(le.classes_)).to(device)
model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model.eval()

# === Collect input ===
def prompt_input():
    print("Available symptoms:")
    for i, s in enumerate(SYMPTOMS, 1):
        print(f" {i}. {s}")
    chosen = input("Enter up to 3 symptom numbers (comma-separated): ").strip()
    chosen_idxs = []
    if chosen:
        chosen_idxs = [int(x.strip()) - 1 for x in chosen.split(",") if x.strip().isdigit()]
    chosen_idxs = [i for i in chosen_idxs if 0 <= i < len(SYMPTOMS)][:3]

    Age = float(input("Age: ").strip())
    Gender = 1 if input("Gender (Male/Female): ").strip().lower() in ["male","m","1"] else 0
    HR = float(input("Heart Rate (bpm): ").strip())
    Temp = float(input("Body Temperature (C): ").strip())
    SpO2 = float(input("Oxygen Saturation (%): ").strip())
    Sys = float(input("Systolic BP: ").strip())
    Dia = float(input("Diastolic BP: ").strip())

    feats = [Age, Gender, HR, Temp, SpO2, Sys, Dia]
    for i in range(len(SYMPTOMS)):
        feats.append(1.0 if i in chosen_idxs else 0.0)
    return np.array(feats, dtype=np.float32)

# === Predict ===
x = prompt_input()
x_scaled = scaler.transform(x.reshape(1,-1))
with torch.no_grad():
    s = torch.FloatTensor(x_scaled).to(device)
    logits = model(s)
    probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()

topN = 3
top_idx = np.argsort(probs)[::-1][:topN]
print("\nPredicted Diseases:")
for i in top_idx:
    print(f" {le.classes_[i]}: {probs[i]*100:.2f}%")

pred_idx = top_idx[0]
pred_label = le.inverse_transform([pred_idx])[0]
print(f"\nMost likely: {pred_label}")

# === Doctor feedback ===
feedback = input("\nIs this correct? (y/n): ").strip().lower()
if feedback == "y":
    reward = 10.0
    true_label = pred_label
else:
    true_label = input("Enter the correct disease: ").strip()
    if true_label not in le.classes_:
        print("Unknown label; skipping update.")
        exit()
    reward = -5.0

true_idx = int(np.where(le.classes_ == true_label)[0][0])

# === Reinforcement update ===
print("\nApplying feedback update (REINFORCE)...")

model.train()
opt = optim.Adam(model.parameters(), lr=1e-5)

s_t = torch.FloatTensor(x_scaled).to(device)
logits = model(s_t)
probs = torch.softmax(logits, dim=1)
m = torch.distributions.Categorical(probs)
action = pred_idx
logp = m.log_prob(torch.tensor(action).to(device))

# Reward signal: positive for correct, negative for incorrect
loss = -logp * reward
opt.zero_grad()
loss.backward()
opt.step()

torch.save(model.state_dict(), BEST_MODEL)
print("Model updated and saved based on feedback.")
