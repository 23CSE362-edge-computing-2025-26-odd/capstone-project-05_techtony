# test_dr_pg.py
# Load saved model and evaluate on saved test split

import numpy as np, joblib, torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch.nn as nn

SCALER_PATH = "scaler_pg.pkl"
ENCODER_PATH = "labelenc_pg.pkl"
BEST_MODEL = "pg_drl_model_best.pth"
X_TEST = "X_test_pg.npy"
Y_TEST = "y_test_pg.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)
X_test = np.load(X_TEST)
y_test = np.load(Y_TEST)

class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

model = ActorNet(X_test.shape[1], len(le.classes_)).to(device)
model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for i in range(len(X_test)):
        s = torch.FloatTensor(X_test[i]).unsqueeze(0).to(device)
        logits = model(s)
        preds.append(int(logits.argmax(1).item()))

acc = accuracy_score(y_test, preds)
print(f"Test Accuracy: {acc*100:.2f}%")
print("Classification report:")
print(classification_report(y_test, preds, target_names=le.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_test, preds))

print("\nExamples:")
for i in range(min(10, len(X_test))):
    print("True:", le.inverse_transform([y_test[i]])[0], " Pred:", le.inverse_transform([preds[i]])[0])
