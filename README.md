# **GraphQL Anomaly Detector**

An AI-driven system for detecting malicious GraphQL queries, with a focus on **deeply nested**, **high-cost**, and **DoS-style attack patterns**.
The system combines **AST analysis**, **query complexity metrics**, **rule-based heuristics**, **synthetic traffic generation**, and **ML-based anomaly detection**.

---

## **Core Goals**

* Detect anomalous or abusive GraphQL queries in real time
* Identify excessive nesting, recursion, alias overuse, and unusually high complexity
* Generate both **benign** and **malicious** queries for ML training
* Integrate detection logic directly into any GraphQL server
* Provide explainable rule-based signals alongside ML scoring

---

## **External Dependencies (Verified Repos)**

Cloned into `/external/`:

| Repo                         | Purpose                                                               |
| ---------------------------- | --------------------------------------------------------------------- |
| **graphql-js**               | AST parsing, schema utilities, base GraphQL execution environment     |
| **graphql-query-complexity** | Cost/depth/complexity scoring for queries                             |
| **graphql-cop**              | Rule-based checks for structural anomalies and common attack patterns |

These are the only third-party components required.
All traffic generation (normal + malicious) is implemented in-house.

---

## **System Architecture Overview**

### **1. Ingestion Layer**

Captures incoming queries through middleware, extracts AST and structural metadata, computes cost/depth metrics, and prepares ML feature vectors.

### **2. Synthetic Data Pipeline**

Generates:

* **Normal queries** based on schema introspection
* **Malicious queries** (nested, recursive, alias-heavy, high-entropy)
* **Traffic sequences** for time-series learning

Includes preprocessing, normalization, windowing, and dataset assembly.

### **3. Machine Learning**

Implements:

* LSTM-based anomaly detector (primary model)
* Optional transformer model
* Static + temporal feature fusion

Includes training code, experiment configuration, and model export utilities.

### **4. Real-Time Detection Engine**

Combines:

* Rule-based scores (GraphQL-Cop)
* ML anomaly scores
* Threshold-based decisioning

Actions include: allow, throttle, or block.

### **5. Integration Layer**

Provides server middleware and monitoring hooks (Prometheus/Grafana).
Supports production deployment via Docker and Kubernetes.

---

## **Key Directories**

```
external/            # verified third-party repos
src/ingestion/       # parsing, feature extraction, publishing
src/data_pipeline/   # generators + preprocessing + datasets
src/ml/              # models, training, export, inference
src/realtime_detector/  # rule+ML fusion and real-time actioning
src/integration/     # server patches, metrics, logging
infra/               # docker, k8s, compose
configs/             # system/model/data config files
tests/               # unit and integration tests
```

---

## **Workflow Summary**

1. **Ingest query** → parse → compute cost/depth → extract features
2. **Send features** to real-time detector
3. **Apply rule engine** (GraphQL-Cop)
4. **Run ML model** for anomaly scoring
5. **Make decision** (allow / throttle / block)
6. **Log** results for retraining
7. **Offline pipeline** builds datasets and trains models with synthetically generated queries

---

## **Status**

This repository defines the architecture, directory layout, and major components required for the GraphQL anomaly detection system.
Implementation of modules will proceed in phases:

* Feature extraction → Data generation → ML training → Real-time engine → Integration.

---
