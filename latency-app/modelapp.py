# # Integration of DRL and FL
# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# import torch
# import torch.nn as nn
# import os
# import requests
# import time
# import threading

# # --- Configuration ---
# SCALER_PATH = "scaler_pg.pkl"
# ENCODER_PATH = "labelenc_pg.pkl"
# MODEL_PATH = "pg_drl_model_best.pth"
# FL_SERVER_URL = "http://host.docker.internal:5000/submit-update" # URL for the FL Server

# DEVICE = torch.device("cpu")
# app = Flask(__name__)

# # --- Global State ---
# model = None
# scaler = None
# le = None
# LOCAL_MODEL_UPDATES = []

# # --- Model Definition & Loading ---
# class ActorNet(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128), nn.ReLU(),
#             nn.Linear(128, 128), nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )
#     def forward(self, x): return self.net(x)

# def load_artifacts():
#     """Loads the DRL model and other artifacts into memory."""
#     global model, scaler, le
#     try:
#         scaler = joblib.load(SCALER_PATH)
#         le = joblib.load(ENCODER_PATH)
#         n_features = len(scaler.mean_)
#         n_classes = len(le.classes_)
#         model = ActorNet(n_features, n_classes).to(DEVICE)
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
#         model.eval()
#         print("--- DRL Model and artifacts loaded successfully! ---")
#     except Exception as e:
#         print(f"FATAL ERROR loading artifacts: {e}")

# # --- Federated Learning Communication ---
# def send_updates_to_fl_server():
#     """Background task to periodically send updates."""
#     while True:
#         time.sleep(30) # Send updates every 30 seconds
#         global LOCAL_MODEL_UPDATES
#         if LOCAL_MODEL_UPDATES:
#             print(f"Simulating sending {len(LOCAL_MODEL_UPDATES)} updates to the FL Server...")
#             try:
#                 # In a real system, you'd send actual gradients or weights.
#                 # Here, we just send a count for the simulation.
#                 payload = {"edge_id": "edge-node-1", "update_count": len(LOCAL_MODEL_UPDATES)}
#                 requests.post(FL_SERVER_URL, json=payload)
#                 LOCAL_MODEL_UPDATES = [] # Clear updates after sending
#             except Exception as e:
#                 print(f"Could not send updates to FL Server: {e}")

# # --- API Endpoint ---
# @app.route('/', methods=['POST'])
# def handle_request():
#     """Receives data from EdgeX and uses the DRL model."""
#     if not model:
#         return jsonify({"status": "error", "message": "Model is not loaded."}), 500

#     try:
#         event_data = request.get_json(force=True)
#         readings = {r['resourceName']: r['value'] for r in event_data['event']['readings']}
        
#         feature_array = np.array([[
#             float(readings['Age']), float(readings['Gender']), float(readings['HeartRate']),
#             float(readings['Temperature']), float(readings['SpO2']), float(readings['SystolicBP']),
#             float(readings['DiastolicBP'])
#         ]], dtype=np.float32)

#         x_scaled = scaler.transform(feature_array)
#         with torch.no_grad():
#             inp = torch.FloatTensor(x_scaled).to(DEVICE)
#             logits = model(inp)
#             pred_idx = logits.argmax(dim=1).item()
#             pred_class = le.inverse_transform([pred_idx])[0]

#         # Simulate generating a "local update" after a successful prediction
#         LOCAL_MODEL_UPDATES.append({"prediction": pred_class})
        
#         print(f"Prediction: {pred_class}. Total local updates: {len(LOCAL_MODEL_UPDATES)}")
#         return jsonify({"status": "success", "predicted_disease": pred_class})

#     except Exception as e:
#         return jsonify({"status": "error", "message": str(e)}), 400

# if __name__ == '__main__':
#     load_artifacts()
#     # Start the background thread for FL communication
#     update_thread = threading.Thread(target=send_updates_to_fl_server, daemon=True)
#     update_thread.start()
#     app.run(host='0.0.0.0', port=8080)
# Integration of DRL and FL
# app.py - DRL Edge Agent with Latency Metrics and FL Communication
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge
import joblib, numpy as np, torch, torch.nn as nn, torch.optim as optim
import os, requests, time, threading, random

# --- Configuration & Global Artifacts ---
SCALER_PATH = "scaler_pg.pkl"
ENCODER_PATH = "labelenc_pg.pkl"
BEST_MODEL = "pg_drl_model_best.pth"
# This should point to the FL Server. If running it in Docker Compose, 'edgex-fl-server' is the hostname.
FL_SERVER_URL = "http://localhost:5001" 
DEVICE = torch.device("cpu")

app = Flask(__name__)
metrics = PrometheusMetrics(app) # For automatic metrics like request counts

# --- Global State ---
model, scaler, le = None, None, None
LOCAL_MODEL_UPDATES = []
response_times = []
MAX_SAMPLES = 20

# --- Custom Prometheus Metric for HPA ---
flask_http_request_duration_seconds_avg = Gauge(
    'flask_http_request_duration_seconds_avg', 
    'Average HTTP request duration for HPA scaling',
    ['method', 'path', 'status']
)

# --- DRL Model Logic ---
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

FEATURE_NAMES = ["Age","Gender","HeartRate","Temperature","SpO2","SystolicBP","DiastolicBP"]

def load_artifacts():
    global model, scaler, le
    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(ENCODER_PATH)
        n_features = len(scaler.mean_)
        n_classes = len(le.classes_)
        model = ActorNet(n_features, n_classes).to(DEVICE)
        model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
        model.eval()
        print("--- DRL Model and artifacts loaded successfully! ---")
    except Exception as e:
        print(f"FATAL ERROR loading artifacts: {e}")

# --- Federated Learning Communication ---
def send_updates_to_fl_server():
    """Background task to periodically send simulated updates."""
    while True:
        time.sleep(30) # Send updates every 30 seconds
        global LOCAL_MODEL_UPDATES
        if LOCAL_MODEL_UPDATES:
            print(f"---  sending {len(LOCAL_MODEL_UPDATES)} updates to the FL Server... ---")
            try:
                payload = {"edge_id": "Hospital-Edge-Node-1", "update_content": LOCAL_MODEL_UPDATES}
                # Use a timeout to prevent blocking
                requests.post(f"{FL_SERVER_URL}/submit-update", json=payload, timeout=5)
                LOCAL_MODEL_UPDATES = [] # Clear updates after sending
            except Exception as e:
                print(f"--- Could not send updates to FL Server: {e} ---")

# --- API Endpoints ---
@app.route('/', methods=['POST'])
def handle_request():
    """Receives data from EdgeX, performs DRL inference, and calculates latency."""
    start_time = time.time()
    
    if not model:
        return jsonify({"status": "error", "message": "Model is not loaded."}), 500

    prediction_result = "error"
    try:
        event_data = request.get_json(force=True)
        readings = {r['resourceName']: r['value'] for r in event_data['event']['readings']}
        
        feature_array = np.array([[
            float(readings['Age']), float(readings['Gender']), float(readings['HeartRate']),
            float(readings['Temperature']), float(readings['SpO2']), float(readings['SystolicBP']),
            float(readings['DiastolicBP'])
        ]], dtype=np.float32)

        # --- DRL INFERENCE ---
        x_scaled = scaler.transform(feature_array)
        with torch.no_grad():
            inp = torch.FloatTensor(x_scaled).to(DEVICE)
            logits = model(inp)
            pred_idx = logits.argmax(dim=1).item()
            pred_class = le.inverse_transform([pred_idx])[0]
        
        prediction_result = pred_class
        # --- SIMULATE LEARNING UPDATE ---
        LOCAL_MODEL_UPDATES.append({"prediction": pred_class, "timestamp": time.time()})
        
    except Exception as e:
        print(f"Error during DRL inference: {e}")
        prediction_result = f"Error: {e}"

    # --- LATENCY CALCULATION & METRIC UPDATE ---
    actual_time = time.time() - start_time
    response_times.append(actual_time)
    if len(response_times) > MAX_SAMPLES:
        response_times.pop(0)
    
    current_avg = sum(response_times) / len(response_times)
    
    # This is the crucial line that updates the metric for the HPA
    flask_http_request_duration_seconds_avg.labels(method='POST', path='/', status='200').set(current_avg)
    
    print(f"Prediction: {prediction_result}. Latency: {actual_time:.3f}s. Avg Latency: {current_avg:.3f}s")
    
    return jsonify({
        "status": "success",
        "predicted_disease": prediction_result,
        "processing_time_seconds": actual_time
    })
@app.route('/receive-global-model', methods=['POST'])
def receive_global_model():
    """Receives global model update from FL Server (downlink)."""
    try:
        data = request.get_json(force=True)
        version = data.get("model_version")
        weights = data.get("weights", [])
        print(f"[⬇️ GLOBAL MODEL RECEIVED] Version {version}, Weights: {weights}")
        return jsonify({"status": "success", "received_version": version})
    except Exception as e:
        print(f"[❌ ERROR receiving global model] {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    load_artifacts()
    # Start the background thread for FL communication
    update_thread = threading.Thread(target=send_updates_to_fl_server, daemon=True)
    update_thread.start()
    app.run(host='0.0.0.0', port=8080)