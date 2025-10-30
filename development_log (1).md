# Development Log

## Project: EdgeX-based Healthcare Monitoring System with Federated Learning and Autoscaling

This log documents the chronological development history based on file modification timestamps.

---

## June 25, 2025

**17:34:53** - Initial docker-compose.yml created  
**17:37:37** - Added project governance files (Contributing.md, .gitignore, semantic.yml, PULL_REQUEST_TEMPLATE.md, GOVERNANCE.md, OWNERS.md, LICENSE)  
**18:44:07** - Set up EdgeX compose builder infrastructure with Makefiles and README files  
**18:44:07** - Added device service configurations for multiple protocols (RFID-LLRP, BACnet, Camera, CoAP, GPIO, Grove, Modbus, MQTT, REST, SNMP, Virtual devices)  
**18:44:07** - Configured Application Service Configurable (ASC) with HTTP and MQTT export options  
**18:44:07** - Set up message bus templates (MQTT, NATS, ZMQ)  
**18:44:07** - Integrated security layers and Redis messagebus configurations  
**18:44:07** - Added Test Automation Framework (TAF) compose files for various architectures (ARM64, x86)  
**18:44:07** - Created environment configuration files (as-common.env, asc-http-export.env, common-security.env, device-common.env)  
**18:44:07** - Added utility scripts (gen_secure_compose_ext.sh, gen_mqtt_messagebus_compose_ext.sh, get-consul-acl-token.sh, upload-api-gateway-cert.sh)  
**19:13:58** - Updated get-api-gateway-token.sh  
**19:15:28** - Configured .env file

---

## September 2, 2025

**01:22:08-01:22:19** - Initialized Git repository with hooks and samples  
**01:22:15** - Cloned EdgeX compose repository with full structure  
**01:22:15** - Duplicated EdgeX configuration files (Contributing.md, PULL_REQUEST_TEMPLATE.md, semantic.yml, .gitignore, GOVERNANCE.md, LICENSE, OWNERS.md)  
**01:22:16** - Added runtime token configuration generation script  
**01:22:17** - Updated Makefiles and README files for both directories  
**01:22:18** - Deployed comprehensive device service compose files:
  - add-app-rfid-llrp-inventory.yml
  - add-asc-external-mqtt-trigger.yml
  - add-asc-http-export.yml
  - add-asc-metrics-influxdb.yml
  - add-asc-mqtt-export.yml
  - add-asc-sample.yml
  - add-delayed-start-services.yml
  - Device services: BACnet, Camera, CoAP, GPIO, Grove, Modbus, MQTT, ONVIF Camera, REST, RFID-LLRP, SNMP, USB Camera, Virtual
  - add-modbus-simulator.yml
  - Message bus configurations: MQTT, NATS
  - Security templates: secure MQTT/Redis messagebus, secure device MQTT
  - TAF configurations for testing
**01:22:18** - Set up environment files (as-common.env, asc-http-export.env, asc-mqtt-export.env, common-sec-stage-gate.env, common-security.env, common.env, device-common.env, mqtt-bus.env, nats-bus.env)  
**01:22:18** - Created docker-compose-base.yml  
**01:22:18-01:22:19** - Added generation scripts and utilities (gen-header, gen_mqtt_messagebus_compose_ext.sh, gen_nats_messagebus_compose_ext.sh, gen_secure_compose_ext.sh, get-api-gateway-token.sh, get-consul-acl-token.sh, tui-generator.sh, upload-api-gateway-cert.sh)  
**01:22:19** - Deployed Docker Compose files for multiple configurations:
  - ARM64 variants
  - No-security variants
  - With app sample variants
  - Portainer integration
  - TAF variants (MQTT bus, no security, performance testing)
**01:22:19** - Set up Git branch structure (levski branch)

---

## September 3, 2025

**02:31:09** - Created Kubernetes app-deployment.yaml  
**02:31:35** - Added Horizontal Pod Autoscaler configuration (app-hpa.yaml)  
**03:00:02** - Developed initial load generator (load_generator.py)  
**03:02:55** - Created edge-autoscaling-project directory  
**03:03:22** - Added IP-based load generator (load_generator_ip.py)

---

## September 6, 2025

**01:02:41** - Updated docker-compose.yml  
**18:47:09** - Created load generator v2 (load_generator_v2.py)  
**19:09:24** - Updated compose-builder

---

## September 8, 2025

**14:01:59** - Modified Git config  
**14:02:04** - Updated .git directory  
**14:45:02** - Deployed production docker-compose.yml  
**14:48:55** - Configured Consul service discovery (consul-config)  
**16:26:10** - Set up eKuiper rules engine data directories:
  - uploads, sources, sinks, functions, schemas, services
  - connection.yaml for connectivity management
  - connections directory
  - sqliteKV.db for persistent storage
  - initialized kuiper-data
  - sqliteKV.db-wal for write-ahead logging

---

## September 11, 2025

**16:51:02** - Created health-monitor-profile.json  
**17:30:30** - Added health-monitor-service.json  
**17:37:57** - Configured health-monitor-device.json

---

## September 12, 2025

**02:11:26** - Added Swagger UI documentation screenshot (swagger-ui.png)  
**02:49:26** - Added Docker architecture diagram (docker.png)  
**02:50:33** - Added EdgeX architecture diagram (edgex.png)  
**02:51:33** - Added Kubernetes architecture diagram (kuber.png)  
**02:52:53** - Added Prometheus monitoring diagram (prometheus.png)  
**02:53:12** - Added Grafana visualization screenshot (grafana.jpeg)  
**02:54:13** - Added HPA diagram (hpa.jpeg)  
**05:34:25** - Configured custom metrics for Prometheus (custom-metrics-values.yaml)

---

## September 13, 2025

**20:09:15** - Created demo.yml for demonstration setup

---

## September 14, 2025

**02:04:25** - Initialized Consul cluster (node-id)  
**02:04:25** - Set up Raft consensus (raft directory)  
**02:04:25** - Configured peer information (peers.info)  
**02:04:25** - Set up Serf gossip protocol (serf directory)  
**02:04:36** - Created checkpoint-signature  
**02:04:37** - Established consul-data persistence  
**02:17:09** - Added device-details.json  
**03:21:22** - Created Docker container data (hash: 61e8e06ebfa5d4ad7688ded71ce8cc3be457239e67cdbe358ddf49360cf6ecbd)  
**04:05:03** - Updated edgex-compose directory  
**20:43:51** - Added screenshot (Screenshot (80).png)

---

## September 15, 2025

**00:25:21** - Created servicemonitor-fix.yaml for Prometheus  
**00:25:24** - Added service.yaml  
**00:29:45** - Configured additional-scrape-config.yaml  
**04:49:20** - Developed Flask application (app.py) for health data processing

---

## September 18, 2025

**00:42:44** - Added architecture diagram (arc.jpg)  
**00:49:43** - Deployed DRL model components:
  - labelenc_pg.pkl (label encoder)
  - pg_drl_model_best.pth (best trained model)
  - pg_drl_model_final.pth (final model)
  - predict.py (prediction script)
  - scaler_pg.pkl (data scaler)
**00:50:41** - Added Federated Learning model (fed_classifier_ae_transfer.tflite)  
**01:41:31** - Created latency-app directory  
**04:03:34-04:04:01** - Added system screenshots (Screenshot (90).png, Screenshot (91).png)  
**04:39:24-04:39:25** - Added experiment images (e1.jpg, e2.jpg, e3.jpg, e4.jpg)  
**04:55:39** - Created demonstration video (rv.mp4)  
**05:01:56-05:39:53** - Produced multiple demo videos:
  - kv - Made with Clipchamp.mp4
  - jv - Made with Clipchamp.mp4
  - jv1 - Made with Clipchamp.mp4
  - kv1.mp4
  - jv2.mp4
**06:14:24-06:14:27** - Added more screenshots (Screenshot (92).png, Screenshot (93).png)  
**10:32:10** - Created narration audio (nvoice.mp3)  
**10:38:28** - Produced narrated video (nv - Made with Clipchamp.mp4)  
**10:47:23** - Developed review2.html web interface

---

## September 19, 2025

**22:05:59** - Updated reviewcheck.html

---

## September 20, 2025

**01:02:52** - Created patient-monitor-profile.json  
**01:12:20** - Added patient-monitor-device.json  
**01:32:39** - Configured patient-monitor-service.json  
**02:44:29** - Updated docker-compose-no-secty.yml  
**18:19:44** - Created Dockerfile for containerization  
**19:19:06** - Added requirements.txt for Python dependencies

---

## September 21, 2025

**12:04:09** - Created Consul snapshot (24-16386-1758436449069)  
**12:04:09** - Generated state.bin and meta.json for cluster state

---

## October 5, 2025

**09:51:56** - Updated SQLite shared memory (sqliteKV.db-shm)  
**09:51:59** - Created remote Consul snapshot (remote.snapshot)  
**09:52:08** - Created local Consul snapshot (local.snapshot)  
**10:54:37** - Generated Redis database dump (dump.rdb)  
**10:55:47** - Updated Raft database (raft.db)  
**10:55:57** - Created temporary Redis file (temp-1.rdb)

---

## October 11, 2025

**11:28:49** - Created comprehensive Architecture.png  
**11:34:25** - Developed review3.html dashboard

---

## October 14, 2025

**04:13:10** - Generated performance analysis (latencyvscpu.html)  
**16:23:51** - Fetched latest updates (FETCH_HEAD)  
**16:24:33** - Deployed complete CI_DRL system:
  - X_test_pg.npy, y_test_pg.npy (test datasets)
  - lab_health.csv, raw_health.csv (health datasets)
  - labelenc_pg.pkl, scaler_pg.pkl (preprocessing artifacts)
  - main_train.py (training script)
  - pg_drl_model_best.pth, pg_drl_model_final.pth (trained models)
  - predict.py (prediction interface)
  - test.py (testing suite)
**16:24:33** - Added Federated Learning components:
  - FL_model directory
  - FL_model.ipynb (Jupyter notebook)
  - FL_model_dataset.md (dataset documentation)
**16:24:33** - Added project documentation:
  - Presentation_file.md
  - Review_2.md
  - edge_computing.pptx
  - sensors-24-01346-v3.pdf (research paper)
**16:25:14** - Reorganized edgex-new directory structure  
**16:25:14** - Restructured Git repository (branches, hooks, info, logs, refs, objects, pack)  
**16:25:14** - Organized project directories:
  - .github, compose-builder, consul-config
  - consul-data with Raft, Serf, snapshots
  - db-data for database persistence
  - kuiper-data with connections, functions, schemas, services, sinks, sources, uploads
  - edgex-compose, taf
**16:26:49** - Updated Git index

---

## October 19, 2025

**14:38:14** - Created arc1.png (architecture diagram)  
**14:48:57** - Generated graphs.png (performance graphs)  
**14:52:42** - Added collab2.png (collaboration diagram)  
**14:55:45** - Added collab1.png  
**15:02:39** - Created collab.jpg  
**15:40:05** - Merged PDF documentation (ilovepdf_merged (1).pdf)

---

## October 20, 2025

**14:48:11** - Generated LaTeX report (Edge_Computing_Report_template_Latex-RJKV.pdf)  
**17:46:47** - Created arc-overview.png (system overview)  
**17:48:40** - Added supplementary documentation (second (1).pdf)  
**17:51:41** - Updated report (Edge_Computing_Report_template_Latex (3).pdf)

---

## October 21, 2025

**17:04:18** - Created HPA v2 configurations (hpa(v2).yaml, hpa.yaml)

---

## October 22, 2025

**02:08:44** - Updated Kubernetes deployments (deployment.yaml, deployment(v2).yaml)  
**02:37:47** - Refined custom metrics configurations (custom-metrics-values(v2).yaml, custom-metrics-values.yaml)  
**20:44:52** - Updated docker-compose-no-secty.yml  
**22:15:29** - Finalized report (Edge_Computing_Report_template_Latex ).pdf)

---

## October 23, 2025

**01:21:13** - Major CI_DRL_v2 update:
  - Enhanced dataset files (X_test_pg.npy, y_test_pg.npy, lab_health.csv, raw_health.csv, symptom_list.csv)
  - Updated preprocessing artifacts (labelenc_pg.pkl, scaler_pg.pkl)
  - Improved training pipeline (main_train.py, prepare_dataset.py)
  - Updated models (pg_drl_model_best.pth, pg_drl_model_final.pth)
  - Enhanced prediction and testing (predict.py, test.py)
**01:21:13** - Updated device configurations (health-monitor-device.json, health-monitor-profile.json, health-monitor-service.json)  
**01:21:13** - Added CPU-based autoscaling (deployment-cpu.yaml, hpa-cpu.yaml)  
**01:21:13** - Updated latency-app with app.py  
**01:21:13** - Configured patient monitoring devices (patient-monitor-device.json, patient-monitor-profile.json, patient-monitor-service.json)  
**01:24:59** - Created service-monitor.yaml  
**01:25:49** - Added servicemonitor.yaml  
**01:25:54** - Created sm-fix.yaml  
**01:26:24-01:26:28** - Configured HPA permissions (hpa-rbac.yaml, hpa-permissions.yaml)  
**01:26:32** - Added direct-scrape.yaml  
**01:26:46** - Established k8s-config directory  
**01:27:25-01:27:35** - Configured Prometheus adapter:
  - prometheus-adapter-config.yaml
  - prometheus-adapter-fixed.yaml
  - prometheus-adapter-values-fixed.yaml
  - prometheus-adapter-values.yaml
**01:27:54** - Added adapter-rbac.yaml  
**01:31:18** - Finalized CI_DRL_v2 directory structure  
**02:05:00** - Created fl_server directory with simulated_global_model.txt  
**02:36:10** - Developed modelapp.py for model serving  
**02:50:30** - Created simulate_patient.ps1 for patient data simulation  
**08:53:04** - Updated SQLite shared memory (sqliteKV.db-shm)  
**08:53:04** - Updated snapshots directory  
**08:53:07** - Created new remote.snapshot  
**08:53:16** - Created new local.snapshot  
**09:32:14** - Updated pg_drl_model_best.pth  
**09:58:11** - Generated new Redis dump (dump.rdb)  
**09:58:11** - Updated db-data  
**10:00:08** - Updated Consul services  
**10:00:08** - Updated raft.db  
**10:00:09** - Updated health checks  
**10:00:21** - Created new temp-1.rdb

---

## October 24, 2025

**10:34:34** - Enhanced fl_server.py (Federated Learning server)  
**10:39:29** - Updated k8s-config directory  
**10:42:09** - Reorganized k8s-config structure  
**10:42:52** - Renamed project to capstone-project-05_techtony  
**10:42:52** - Finalized fl_server directory  
**10:43:35** - Generated CI reports (cireports .pdf)  
**10:44:04** - Updated modelapp.py  
**11:00:05** - Created readme documentation (readme (2).md)  
**11:06:46** - Organized documents directory  
**11:12:09** - Updated main README.md  
**11:12:36** - Updated .git directory  
**11:40:48** - Final repository synchronization (FETCH_HEAD)  
**11:40:48** - Optimized Git objects

---

## Project Summary

**Total Development Duration**: June 25, 2025 - October 24, 2025 (4 months)

**Key Milestones**:
- EdgeX Foundry integration with 15+ device protocols
- Deep Reinforcement Learning for clinical intelligence
- Federated Learning implementation for privacy-preserving AI
- Kubernetes autoscaling with custom metrics
- Complete monitoring stack (Prometheus, Grafana, Consul)
- Comprehensive documentation and demonstration materials

**Final Status**: Production-ready healthcare monitoring system with edge AI capabilities