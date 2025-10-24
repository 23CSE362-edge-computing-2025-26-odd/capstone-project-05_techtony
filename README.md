[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/WzUeh8r0)
# A Scalable Edge-Cloud Framework for Real-Time Remote Patient Monitoring

This project demonstrates a complete, end-to-end framework for real-time patient monitoring using a distributed AI architecture. The system ingests simulated IoT data at the edge using **EdgeX Foundry**, processes it with a **Deep Reinforcement Learning (DRL)** model running in a **Kubernetes** cluster, and simulates a **Federated Learning (FL)** loop for collaborative, privacy-preserving model improvement.

A key feature is the implementation of dynamic, performance-based autoscaling, allowing the system to scale based on either **CPU load** or **application latency**.

---

## 1. System Architecture

The architecture is divided into three logical layers: the Edge, the Edge Cluster, and the Central Cloud.

- **Edge (Data Ingestion)**: An EdgeX Foundry stack running in Docker simulates IoT devices, ingests data, and forwards it for processing.

- **Edge Cluster (Real-Time Processing)**: A Kubernetes cluster, also running at the edge, hosts the DRL application, the monitoring stack, and the autoscaling components.

- **Central Cloud (Collaborative Learning)**: A central server manages the Federated Learning process by aggregating model updates and distributing improved global models.

### End-to-End Workflow

<img width="1620" height="931" alt="image" src="https://github.com/user-attachments/assets/e3d5777a-10e2-472a-9e43-2da5d7f9cdb7" />


---

## 2. Technologies Used

| Category | Technology | Purpose |
|----------|------------|---------|
| **IoT & Edge** | EdgeX Foundry | Open-source IoT middleware for data ingestion and processing at the edge. |
| **Containerization** | Docker, Docker Compose | To package and run all EdgeX and FL Server services consistently. |
| **Orchestration** | Kubernetes | To manage, deploy, and automatically scale the DRL application at the edge. |
| **Monitoring** | Prometheus, Grafana | For collecting time-series metrics (latency, CPU) and visualizing them in real-time dashboards. |
| **AI/ML** | PyTorch, TensorFlow Lite | To build, train, and run the DRL model (edge) and the Global model (cloud). |
| **Application** | Python, Flask | The framework for the DRL edge agent and the central FL server. |

---

## 3. Setup and Installation

### Prerequisites

- Docker Desktop with Kubernetes enabled
- `kubectl` command-line tool
- `helm` for managing Kubernetes packages

### Step 1: Set Up the EdgeX Environment

1. **Register EdgeX Components**:

```powershell
# 1. Register Device Profile
curl http://localhost:59881/api/v2/deviceprofile -H "Content-Type: application/json" -d '@path/to/patient-monitor-profile.json'

# 2. Register Device Service
curl http://localhost:59881/api/v2/deviceservice -H "Content-Type: application/json" -d '@path/to/patient-monitor-service.json'

# 3. Register Device
curl http://localhost:59881/api/v2/device -H "Content-Type: application/json" -d '@path/to/patient-monitor-device.json'
```

2. **Start EdgeX Services**:

```powershell
docker-compose -f docker-compose-no-secty.yml up -d --build
```

### Step 2: Set Up the Kubernetes Environment

1. **Build and Push Your Application Image**:
   - Navigate to the `latency-app` directory
   - Build the image: `docker build -t your-dockerhub-username/latency-app:v1 .`
   - Push the image: `docker push your-dockerhub-username/latency-app:v1`

2. **Deploy to Kubernetes**:
   - Navigate to the `k8s-config` directory
   - Update the `image:` in `deployment.yaml` to point to the image you just pushed

```powershell
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Step 3: Configure Autoscaling (CPU or Latency)

#### For CPU-Based Scaling:

1. **Configure HPA for CPU**:

```yaml
# hpa.yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 80
```

2. **Apply the HPA**:

```powershell
kubectl apply -f hpa.yaml
```

#### For Latency-Based Scaling (Advanced):

1. **Install Prometheus Adapter**:

```powershell
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  -f custom-metrics-values.yaml \
  --set prometheus.url=http://monitoring-kube-prometheus-prometheus.monitoring.svc
```

2. **Configure HPA for Latency**:

```yaml
# hpa.yaml
metrics:
- type: Pods
  pods:
    metric:
      name: flask_http_request_duration_seconds_avg
    target:
      type: AverageValue
      averageValue: "500"
```

3. **Apply the HPA**:

```powershell
kubectl apply -f hpa.yaml
```

---

## 4. Usage and Demonstration

### Watching the System in Action

1. **Start the Data Flow**: Ensure your EdgeX stack is running with `docker-compose up -d`

2. **Monitor HPA Status**:

```powershell
kubectl get hpa -w local-target-app-hpa
```

3. **Watch Pods Scale**:

```powershell
kubectl get pods -w
```

4. **View Grafana Dashboards**:

```powershell
# Forward the port
kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring

# Open http://localhost:3000 in your browser
```

As the load from EdgeX increases the CPU or latency of your application, you will see the HPA detect the change and increase the number of REPLICAS, and new pods will be created in real-time.

---

## 5. Troubleshooting Common Issues

### HPA shows `<unknown>`

This means the HPA cannot get the metric.

- **For CPU**: Ensure the Kubernetes Metrics Server is running (`kubectl get pods -n kube-system`)
- **For Latency**: This is a complex issue. Check the Prometheus Adapter logs, ensure its rules in `custom-metrics-values.yaml` are correct (especially the job label), and verify the HPA has the right RBAC permissions.

### Grafana shows "No Data"

This means Prometheus is not scraping your application.

- Ensure your `deployment.yaml` has the correct `prometheus.io/scrape: "true"` annotations
- If using the Prometheus Operator, ensure you have a ServiceMonitor manifest that correctly targets your application's service

### EdgeX Services are Unhealthy

- Check for port conflicts on your host machine
- Ensure all necessary environment variables are set in `docker-compose-no-secty.yml`
- Use `docker-compose logs` to debug the specific service that is failing

---

## Contributing

Please feel free to submit issues and pull requests to improve this project.

## License

This project is provided as-is for educational and research purposes.
