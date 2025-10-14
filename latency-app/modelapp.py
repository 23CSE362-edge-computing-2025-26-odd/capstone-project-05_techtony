# Integration of DRL and FL
from flask import Flask, request, jsonify
import joblib
import numpy as np
import torch
import torch.nn as nn
import os
import requests
import time
import threading

# --- Configuration ---
SCALER_PATH = "scaler_pg.pkl"
ENCODER_PATH = "labelenc_pg.pkl"
MODEL_PATH = "pg_drl_model_best.pth"
FL_SERVER_URL = "http://host.docker.internal:5000/submit-update" # URL for the FL Server

DEVICE = torch.device("cpu")
app = Flask(__name__)

# --- Global State ---
model = None
scaler = None
le = None
LOCAL_MODEL_UPDATES = []

# --- Model Definition & Loading ---
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

def load_artifacts():
    """Loads the DRL model and other artifacts into memory."""
    global model, scaler, le
    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(ENCODER_PATH)
        n_features = len(scaler.mean_)
        n_classes = len(le.classes_)
        model = ActorNet(n_features, n_classes).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("--- DRL Model and artifacts loaded successfully! ---")
    except Exception as e:
        print(f"FATAL ERROR loading artifacts: {e}")

# --- Federated Learning Communication ---
def send_updates_to_fl_server():
    """Background task to periodically send updates."""
    while True:
        time.sleep(30) # Send updates every 30 seconds
        global LOCAL_MODEL_UPDATES
        if LOCAL_MODEL_UPDATES:
            print(f"Simulating sending {len(LOCAL_MODEL_UPDATES)} updates to the FL Server...")
            try:
                # In a real system, you'd send actual gradients or weights.
                # Here, we just send a count for the simulation.
                payload = {"edge_id": "edge-node-1", "update_count": len(LOCAL_MODEL_UPDATES)}
                requests.post(FL_SERVER_URL, json=payload)
                LOCAL_MODEL_UPDATES = [] # Clear updates after sending
            except Exception as e:
                print(f"Could not send updates to FL Server: {e}")

# --- API Endpoint ---
@app.route('/', methods=['POST'])
def handle_request():
    """Receives data from EdgeX and uses the DRL model."""
    if not model:
        return jsonify({"status": "error", "message": "Model is not loaded."}), 500

    try:
        event_data = request.get_json(force=True)
        readings = {r['resourceName']: r['value'] for r in event_data['event']['readings']}
        
        feature_array = np.array([[
            float(readings['Age']), float(readings['Gender']), float(readings['HeartRate']),
            float(readings['Temperature']), float(readings['SpO2']), float(readings['SystolicBP']),
            float(readings['DiastolicBP'])
        ]], dtype=np.float32)

        x_scaled = scaler.transform(feature_array)
        with torch.no_grad():
            inp = torch.FloatTensor(x_scaled).to(DEVICE)
            logits = model(inp)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = le.inverse_transform([pred_idx])[0]

        # Simulate generating a "local update" after a successful prediction
        LOCAL_MODEL_UPDATES.append({"prediction": pred_class})
        
        print(f"Prediction: {pred_class}. Total local updates: {len(LOCAL_MODEL_UPDATES)}")
        return jsonify({"status": "success", "predicted_disease": pred_class})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    load_artifacts()
    # Start the background thread for FL communication
    update_thread = threading.Thread(target=send_updates_to_fl_server, daemon=True)
    update_thread.start()
    app.run(host='0.0.0.0', port=8080)