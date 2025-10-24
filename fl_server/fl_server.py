# fl_server.py - Federated Learning Server (Uplink + Downlink Simulation)
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Gauge
import threading, time, json, requests, random

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# --- Prometheus Metrics ---
update_counter = Counter('fl_server_received_updates_total', 'Total number of updates received from edges')
connected_edges = Gauge('fl_server_connected_edges', 'Number of active edge nodes')
last_update_time = Gauge('fl_server_last_update_timestamp', 'Timestamp of last update received')

# --- Global State ---
EDGE_UPDATES = {}
REGISTERED_EDGES = set()
LOCK = threading.Lock()

@app.route('/', methods=['GET'])
def home():
    """Basic server status endpoint."""
    return jsonify({
        "message": "Federated Learning Server active",
        "registered_edges": list(REGISTERED_EDGES),
        "total_updates": sum(len(u) for u in EDGE_UPDATES.values())
    })

# --- Uplink: Edge -> Server ---
@app.route('/submit-update', methods=['POST'])
def receive_update():
    """Receive simulated updates from edges."""
    try:
        data = request.get_json(force=True)
        edge_id = data.get("edge_id", "unknown")
        updates = data.get("update_content", [])

        with LOCK:
            REGISTERED_EDGES.add(edge_id)
            if edge_id not in EDGE_UPDATES:
                EDGE_UPDATES[edge_id] = []
            EDGE_UPDATES[edge_id].extend(updates)
            update_counter.inc(len(updates))
            connected_edges.set(len(REGISTERED_EDGES))
            last_update_time.set(time.time())

        print(f"[✅ RECEIVED] {len(updates)} updates from {edge_id}. Total = {sum(len(u) for u in EDGE_UPDATES.values())}")

        return jsonify({
            "status": "success",
            "message": f"Received {len(updates)} updates from {edge_id}",
            "global_model_version": int(time.time())
        }), 200

    except Exception as e:
        print(f"[❌ ERROR receiving update] {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

# --- Downlink: Server -> Edge ---
def broadcast_global_model():
    """Simulate global model aggregation and push back to all edges."""
    while True:
        time.sleep(90)  # Broadcast every 1.5 minutes

        with LOCK:
            total_updates = sum(len(u) for u in EDGE_UPDATES.values())
            if total_updates == 0:
                continue  # Nothing to aggregate yet
            EDGE_UPDATES.clear()

        global_version = int(time.time())
        model_weights = [round(random.uniform(-1, 1), 4) for _ in range(5)]  # Simulated weights

        print(f"[🌐 BROADCAST] Sending Global Model (v{global_version}) to {len(REGISTERED_EDGES)} edges...")

        for edge_id in REGISTERED_EDGES:
            try:
                # You can replace this with your actual edge IP if known.
                # If all edges run locally, use localhost with different ports.
                edge_url = f"http://localhost:8080/receive-global-model"
                payload = {
                    "model_version": global_version,
                    "weights": model_weights,
                    "timestamp": time.time()
                }
                requests.post(edge_url, json=payload, timeout=3)
                print(f"   ↳ Sent to {edge_id}")
            except Exception as e:
                print(f"[⚠️ DOWNLINK ERROR] Could not send to {edge_id}: {e}")

# --- Manual aggregation endpoint ---
@app.route('/aggregate', methods=['POST'])
def aggregate_models():
    """Simulate model aggregation and return total count."""
    with LOCK:
        total = sum(len(u) for u in EDGE_UPDATES.values())
        EDGE_UPDATES.clear()
    print(f"[🧠 AGGREGATION] Aggregated {total} updates from all edges.")
    return jsonify({"status": "success", "aggregated_updates": total})

# --- Run background threads ---
if __name__ == '__main__':
    threading.Thread(target=broadcast_global_model, daemon=True).start()
    app.run(host='0.0.0.0', port=5001)


# # # # from flask import Flask, request, jsonify, send_file
# # # # import os

# # # # app = Flask(__name__)

# # # # # The Global Model that the server manages and distributes
# # # # GLOBAL_MODEL_PATH = "fed_classifier_ae_transfer.tflite"

# # # # @app.route('/submit-update', methods=['POST'])
# # # # def submit_update():
# # # #     """Receives simulated updates from an edge DRL agent."""
# # # #     data = request.get_json()
# # # #     edge_id = data.get('edge_id', 'unknown')
# # # #     count = data.get('update_count', 0)
# # # #     print(f"Received {count} simulated updates from edge device: {edge_id}")
    
# # # #     # TODO: In a real system, this is where you would trigger
# # # #     # the federated aggregation logic.
    
# # # #     return jsonify({"status": "success", "message": f"Updates from {edge_id} received"})

# # # # @app.route('/get-model', methods=['GET'])
# # # # def get_model():
# # # #     """Allows edge agents to download the latest global model."""
# # # #     print(f"Serving global model '{GLOBAL_MODEL_PATH}' to an edge device.")
# # # #     try:
# # # #         return send_file(GLOBAL_MODEL_PATH, as_attachment=True)
# # # #     except FileNotFoundError:
# # # #         return jsonify({"status": "error", "message": "Global model not found."}), 404

# # # # if __name__ == '__main__':
# # # #     app.run(host='0.0.0.0', port=5000)
# # # # fl_server.py
# # # from flask import Flask, request, jsonify, send_file
# # # import os

# # # app = Flask(__name__)

# # # # The Global Model that the server manages and distributes
# # # GLOBAL_MODEL_PATH = "fed_classifier_ae_transfer.tflite"

# # # @app.route('/submit-update', methods=['POST'])
# # # def submit_update():
# # #     """Receives simulated 'learned weights' from an edge DRL agent."""
# # #     data = request.get_json()
# # #     edge_id = data.get('edge_id', 'unknown')
# # #     update_content = data.get('update_content', {})

# # #     # --- SIMULATION ---
# # #     # In a real system, this is where you would trigger the federated aggregation logic.
# # #     print(f"Received update from edge device: {edge_id}. Content: {update_content}")
# # #     print("Simulating aggregation of weights into the global model...")
# # #     # --- END SIMULATION ---

# # #     return jsonify({"status": "success", "message": f"Updates from {edge_id} received and are being processed."})

# # # @app.route('/get-model', methods=['GET'])
# # # def get_model():
# # #     """Allows edge agents to download the latest global model."""
# # #     print(f"An edge device is requesting the global model '{GLOBAL_MODEL_PATH}'.")
# # #     try:
# # #         return send_file(GLOBAL_MODEL_PATH, as_attachment=True)
# # #     except FileNotFoundError:
# # #         return jsonify({"status": "error", "message": "Global model not found."}), 404

# # # if __name__ == '__main__':
# # #     # Port 5001 is used to avoid conflicts if you run other services locally
# # #     app.run(host='0.0.0.0', port=5001)
# # # fl_server.py
# # from flask import Flask, request, jsonify
# # import time,os

# # app = Flask(__name__)

# # # --- SIMULATION ONLY ---
# # # In a real system, this would be a machine learning model.
# # # Here, it's just a text file that we'll use to prove the update worked.
# # GLOBAL_MODEL_FILE = "simulated_global_model.txt"

# # @app.route('/submit-update', methods=['POST'])
# # def submit_update():
# #     """Receives simulated weights/updates from an edge DRL agent."""
# #     data = request.get_json()
# #     edge_id = data.get('edge_id', 'unknown')

# #     print(f"\n--- Received update from edge device: {edge_id} ---")
# #     print("Simulating aggregation of weights into the global model...")

# #     # --- SIMULATION ---
# #     # We'll update a text file to prove the server processed the update.
# #     with open(GLOBAL_MODEL_FILE, "w") as f:
# #         f.write(f"This is the new global model, updated after learning from {edge_id} at {time.ctime()}")

# #     print("Global model has been improved.")
# #     # --- END SIMULATION ---

# #     return jsonify({"status": "success", "message": "Updates received and processed."})

# # @app.route('/get-model', methods=['GET'])
# # def get_model():
# #     """Allows edge agents to download the latest global model."""
# #     print(f"\n--- Edge device is requesting the global model. ---")
# #     try:
# #         return jsonify({"model_version": time.ctime(), "content": open(GLOBAL_MODEL_PATH).read()})
# #     except FileNotFoundError:
# #         return jsonify({"status": "error", "message": "Global model not found."}), 404

# # if __name__ == '__main__':
# #     # Create a dummy model file on startup
# #     if not os.path.exists(GLOBAL_MODEL_FILE):
# #         with open(GLOBAL_MODEL_FILE, "w") as f:
# #             f.write("This is the initial global model (Version 1).")

# #     # Run on a different port like 5001 to avoid conflicts
# #     app.run(host='0.0.0.0', port=5001)
# from flask import Flask, request, jsonify, send_file
# import time, os, json

# app = Flask(__name__)

# # Global model file simulation
# GLOBAL_MODEL_FILE = "simulated_global_model.txt"

# @app.route('/')
# def home():
#     return jsonify({"status": "running", "message": "FL Server is active"})

# @app.route('/submit-update', methods=['POST'])
# def submit_update():
#     """Receives simulated updates (weights or gradients) from edge devices."""
#     try:
#         data = request.get_json(force=True)
#         edge_id = data.get('edge_id', 'unknown')
#         update_content = data.get('update_content', {})

#         print(f"\n--- Received update from edge device: {edge_id} ---")
#         print(f"Update content: {json.dumps(update_content, indent=2)}")
#         print("Simulating aggregation of weights into the global model...")

#         # --- Simulation of model update ---
#         with open(GLOBAL_MODEL_FILE, "w") as f:
#             f.write(f"Global model updated after receiving from {edge_id} at {time.ctime()}\n")
#             f.write(json.dumps(update_content, indent=2))
#         # --- End simulation ---

#         return jsonify({
#             "status": "success",
#             "message": f"Update from {edge_id} received and global model updated.",
#             "timestamp": time.ctime()
#         })
#     except Exception as e:
#         print("Error during update:", e)
#         return jsonify({"status": "error", "message": str(e)}), 500

# @app.route('/get-model', methods=['GET'])
# def get_model():
#     """Allows edge devices to download the latest global model."""
#     print("\n--- Edge device requested global model ---")
#     try:
#         if not os.path.exists(GLOBAL_MODEL_FILE):
#             raise FileNotFoundError
#         with open(GLOBAL_MODEL_FILE, "r") as f:
#             model_content = f.read()
#         return jsonify({
#             "status": "success",
#             "model_version": time.ctime(os.path.getmtime(GLOBAL_MODEL_FILE)),
#             "content": model_content
#         })
#     except FileNotFoundError:
#         return jsonify({"status": "error", "message": "Global model not found."}), 404

# if __name__ == '__main__':
#     # Ensure model file exists
#     if not os.path.exists(GLOBAL_MODEL_FILE):
#         with open(GLOBAL_MODEL_FILE, "w") as f:
#             f.write("Initial global model created at " + time.ctime())

#     # Run Flask on all interfaces so Docker & Edge can reach it
#     print("\n✅ Federated Learning Server started on port 5001")
#     app.run(host='0.0.0.0', port=5001, debug=True)
