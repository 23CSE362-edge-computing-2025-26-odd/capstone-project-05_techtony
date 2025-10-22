# from flask import Flask, request, jsonify
# from prometheus_flask_exporter import PrometheusMetrics
# import time
# import random

# # Create Flask app
# app = Flask(__name__)

# # Add Prometheus metrics to Flask app
# metrics = PrometheusMetrics(app)

# @app.route('/', methods=['POST'])
# def handle_request():
#     # Simulate a random processing time to generate latency
#     processing_time = random.uniform(0.2, 0.8)
#     time.sleep(processing_time)

#     # Get the incoming data from EdgeX
#     data = request.get_json(force=True)

#     response = {
#         "status": "success",
#         "message": "Data received",
#         "processing_time_seconds": processing_time
#     }
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge
import time
import random

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Custom gauge for the exact metric name your HPA expects
flask_http_request_duration_seconds_avg = Gauge(
    'flask_http_request_duration_seconds_avg', 
    'Average HTTP request duration for HPA scaling',
    ['method', 'path', 'status']
)

# Track response times for rolling average
response_times = []
MAX_SAMPLES = 20  # Keep last 20 samples

@app.route('/', methods=['GET'])
def health_check():
    current_avg = sum(response_times) / len(response_times) if response_times else 0
    return jsonify({
        "status": "healthy", 
        "message": "Flask app is running",
        "current_avg_response_time": current_avg,
        "total_requests": len(response_times)
    })

@app.route('/', methods=['POST'])
def handle_request():
    start_time = time.time()
    
    # Simulate processing time
    processing_time = random.uniform(0.2, 0.8)
    time.sleep(processing_time)
    
    # Calculate actual response time
    actual_time = time.time() - start_time
    
    # Update rolling average
    response_times.append(actual_time)
    if len(response_times) > MAX_SAMPLES:
        response_times.pop(0)
    
    # Update the custom metric with current average
    current_avg = sum(response_times) / len(response_times)
    flask_http_request_duration_seconds_avg.labels(
        method='POST', 
        path='/', 
        status='200'
    ).set(current_avg)
    
    # Process the request
    data = request.get_json(force=True)
    
    response = {
        "status": "success",
        "message": "Data received",
        "processing_time_seconds": actual_time,
        "current_average": current_avg
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)