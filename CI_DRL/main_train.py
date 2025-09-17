# main_train_dr_pg.py
# DRL pipeline: supervised warm-start (Focal Loss) -> REINFORCE policy-gradient fine-tune
# Works on lab_health.csv (preprocessed)

import os, random
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------- CONFIG ----------
CSV_PATH = "lab_health.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Pretraining (supervised)
PRETRAIN_EPOCHS = 60
PRETRAIN_BATCH = 64
PRETRAIN_LR = 1e-3
EARLY_STOP_PATIENCE = 8

# Policy gradient (REINFORCE)
PG_EPISODES = 200
STEPS_PER_EP = 200
PG_LR = 1e-4
GAMMA = 0.99
REWARD_WRONG = -2.0
VALID_EVERY = 10
BASELINE_MOM = 0.99

# Artifacts
SCALER_PATH = "scaler_pg.pkl"
ENCODER_PATH = "labelenc_pg.pkl"
BEST_MODEL = "pg_drl_model_best.pth"
FINAL_MODEL = "pg_drl_model_final.pth"
X_TEST_PATH = "X_test_pg.npy"
Y_TEST_PATH = "y_test_pg.npy"

print("Device:", DEVICE)

# ---------- Load CSV ----------
df = pd.read_csv(CSV_PATH)

if "Disease" not in df.columns:
    raise ValueError("CSV must contain a 'Disease' column")

# features (all except Disease)
feature_cols = [c for c in df.columns if c != "Disease"]
print("Using features:", feature_cols)

X = df[feature_cols].astype(float).values.astype(np.float32)
y = df["Disease"].astype(str).values

# ---------- encode labels and split ----------
le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, ENCODER_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=SEED, stratify=y_enc
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, SCALER_PATH)

np.save(X_TEST_PATH, X_test)
np.save(Y_TEST_PATH, y_test)

state_dim = X_train.shape[1]
n_actions = len(le.classes_)
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}, Classes: {le.classes_}")

# ---------- Actor network ----------
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)  # logits

# ---------- Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1-pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ---------- supervised warm-start ----------
def evaluate_model(model, X, y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X)):
            s = torch.FloatTensor(X[i]).unsqueeze(0).to(DEVICE)
            logits = model(s)
            preds.append(int(logits.argmax(dim=1).item()))
    return accuracy_score(y, preds)

def supervised_warmstart(model, X_tr, y_tr, X_val, y_val):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    crit = FocalLoss(alpha=1, gamma=2)
    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    loader = DataLoader(ds, batch_size=PRETRAIN_BATCH, shuffle=True)

    best_val, best_state, wait = 0.0, None, 0
    for ep in range(1, PRETRAIN_EPOCHS+1):
        model.train()
        total, correct = 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_acc = correct / total
        val_acc = evaluate_model(model, X_val, y_val)
        if ep % 5 == 0 or ep == 1:
            print(f"[Pretrain] Ep {ep}/{PRETRAIN_EPOCHS} TrainAcc {train_acc:.3f} ValAcc {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print(f"[Pretrain] Early stop at epoch {ep}, best val {best_val:.3f}")
                break
    if best_state: model.load_state_dict(best_state)
    return model

# ---------- dataset-driven env with class-based rewards ----------
class DatasetEnv:
    def __init__(self, X, y, label_encoder):
        self.X, self.y, self.n = X, y, len(X)
        self.le = label_encoder
        self.CLASS_REWARDS = {
            "Bronchitis": 10.0,
            "Cold": 10.0,
            "Flu": 15.0,
            "Healthy": 8.0,
            "Pneumonia": 20.0
        }
    def reset(self):
        self.idx = random.randint(0, self.n-1)
        return self.X[self.idx]
    def step(self, action):
        true = self.y[self.idx]
        true_label = self.le.inverse_transform([true])[0]
        if action == true:
            reward = self.CLASS_REWARDS.get(true_label, 10.0)
        else:
            reward = REWARD_WRONG
        self.idx = (self.idx + 1) % self.n
        return self.X[self.idx], reward, False

# ---------- REINFORCE fine-tune ----------
def reinforce_finetune(actor, X_tr, y_tr, X_val, y_val, le):
    actor.to(DEVICE)
    opt = optim.Adam(actor.parameters(), lr=PG_LR)
    env = DatasetEnv(X_tr, y_tr, le)
    running_baseline, best_val = 0.0, 0.0

    for ep in range(1, PG_EPISODES+1):
        state, log_probs, rewards = env.reset(), [], []
        for t in range(STEPS_PER_EP):
            s_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(actor(s_t), dim=1)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()
            log_probs.append(m.log_prob(torch.tensor(action).to(DEVICE)))
            next_state, r, _ = env.step(action)
            rewards.append(r)
            state = next_state

        # returns
        returns, R = [], 0.0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

        # baseline
        running_baseline = BASELINE_MOM * running_baseline + (1-BASELINE_MOM) * returns.mean().item()
        baseline_tensor = torch.tensor(running_baseline).to(DEVICE)
        advantages = returns - baseline_tensor

        # policy loss
        policy_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()
        opt.zero_grad(); policy_loss.backward(); opt.step()

        if ep % VALID_EVERY == 0 or ep == 1:
            val_acc = evaluate_model(actor, X_val, y_val)
            print(f"[PG] Ep {ep}/{PG_EPISODES} ValAcc {val_acc:.3f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(actor.state_dict(), BEST_MODEL)
                print(f"  Saved new best model {best_val:.3f}")

    torch.save(actor.state_dict(), FINAL_MODEL)
    print("Saved final actor:", FINAL_MODEL)
    return actor

# ---------- main ----------
actor = ActorNet(state_dim, n_actions)
print("Starting supervised warm-start...")
actor = supervised_warmstart(actor, X_train, y_train, X_test, y_test)
print("Starting REINFORCE fine-tuning...")
actor = reinforce_finetune(actor, X_train, y_train, X_test, y_test, le)
print("Done.")
