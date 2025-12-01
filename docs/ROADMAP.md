# ✅ **PHASE 1 — Core Foundations (Parse → Metrics → Features)**

### **1. Implement AST Parsing (`ast_extractor.js`)**

**Why first:**
Everything else (normal generator, malicious generator, ML features, rule-based scoring) depends on your ability to parse a GraphQL query and understand its structure.

**What you implement:**

* Parse raw query → AST
* Walk nodes, collect fields, arguments, selections
* Output node-level metadata

**Rationale:**
AST parsing is the **lowest-level primitive**.
Without it, you cannot measure depth, cost, or build feature vectors.

---

### **2. Implement Depth & Cost Calculation (`depth_calc.js`, `cost_calc.js`)**

**Why second:**
Your ML model relies on **structural metrics** (depth, branching, complexity).
Your malicious generator also requires knowing how “deep” or “expensive” a query is.

**What you implement:**

* Max depth
* Average depth
* Branching factor
* Node count
* Cost score using `graphql-query-complexity`

**Rationale:**
These metrics are core **features**.
You cannot generate meaningful training data (normal or malicious) without them.

---

### **3. Implement Feature Extraction (`feature_extractor.py`)**

**Why now:**
This defines your **final ML input schema**, which the entire training pipeline depends on.

**What you implement:**

* Convert AST + metrics → numerical feature vector
* Ensure consistent fields across datasets
* Add entropy, size, alias count, fragment count
* Output JSON or numpy vector

**Rationale:**
You must lock down the **feature format** before generating data or writing ML code.

---

# 🔥 After Phase 1:

You now have:

* AST parsing
* cost/depth metrics
* feature extraction

This means the next major step—**data generation**—is unblocked.

---

# ✅ **PHASE 2 — Data Generation (Normal + Malicious + Sequences)**

## **4. Implement Normal Query Generator (`generate_normal.py`)**

**Why next:**
You need benign traffic before malicious traffic.
This defines the “normal manifold” that ML models must learn.

**What you implement:**

* Schema introspection
* Select random fields
* Generate valid queries with depth 1–4
* Optional arguments
* Mix queries of different types

**Rationale:**
Normal queries define your baseline.
You cannot build anomalies before you understand “normal”.

---

## **5. Implement Malicious Query Generator (`generate_malicious.py`)**

**Why now:**
Now that normal patterns exist, generate structured anomalies that deviate from them.

**What you implement:**

* Deep recursion (depth 10–100)
* Alias bombing
* Fragment recursion
* Over-wide branching
* High entropy names
* Mixed malformed structures (still structurally valid)

**Rationale:**
Your anomaly model must see **clear, controlled malicious variants** of normal patterns.
These become your **positive labels (1)**.

---

## **6. Implement Sequence Generator (`generate_sequences.py`)**

**Why next:**
Your ML includes **LSTM / Transformer** models that require **sequences** not individual queries.

**What you implement:**

* Time-series windows of:

  * normal → normal → normal
  * normal → malicious → malicious
  * mixed bursts
  * slow-burning attacks
* Output sequence of feature arrays

**Rationale:**
Real attackers don’t send one query—they send patterns.
Your model must learn *temporal anomalies*, not just structural ones.

---

### **7. Combine & Label Datasets (`combine_datasets.py`)**

**Why:**
You now generate a dataset with:

* benign
* malicious
* sequences
* mixed traffic

**Rationale:**
Before ML training begins, you need a **unified dataset definition**.

---

# 🔥 After Phase 2:

You now have a full, labeled training set ready for ML.

---

# ✅ **PHASE 3 — ML Pipeline (Training → Evaluation → Export)**

## **8. Build the ML models (`ml/models/*`)**

**Why now:**
All inputs to ML (feature vectors, sequences, labels) are ready.

**What you implement:**

* LSTM anomaly detector
* Transformer (optional)
* Static + temporal ensemble head

**Rationale:**
This is the **first moment** ML development is possible without guessing or changing core inputs.

---

## **9. Build Training Pipeline (`train.py`)**

**What you implement:**

* dataloader
* batching
* loss function
* metrics
* checkpoints
* validation loop

**Rationale:**
A stable training loop is required before the model is usable for real-time inference.

---

## **10. Export Model (`export_torchscript.py` / `export_onnx.py`)**

**Rationale:**
TorchScript or ONNX is required for low-latency real-time inference in production.

---

# 🔥 After Phase 3:

You now have a fully trained model ready for live traffic.

---

# ✅ **PHASE 4 — Real-Time Detection Engine**

## **11. Implement Rule Engine (`rule_engine.py`)**

Uses `graphql-cop`.

**Rationale:**
Rules give low-latency deterministic signals to combine with ML.
This reduces false negatives dramatically.

---

## **12. Implement Scoring Logic (`scoring.py`)**

**Rationale:**
Combines:

* rule-based severity
* ML anomaly score
* pre-configured thresholds

This creates the final risk number.

---

## **13. Implement Real-Time Service (`service.py`)**

**What it does:**

* consumes features
* runs scoring
* triggers allow/throttle/block

**Rationale:**
This ties together inbound features and outbound decisions.

---

# 🔥 After Phase 4:

You have **offline ML training + online detection** working.

---

# ✅ **PHASE 5 — Integration Into GraphQL Server**

## **14. Implement `express_middleware.js`**

**What it does:**

* captures raw query
* extracts features
* sends to real-time detector
* enforces allow/throttle/block
* logs results

**Rationale:**
This is the final cog that puts your ML model into production.

---

# 🔥 After Phase 5:

Your system is **end-to-end functional**.

---

# 📌 **THE FINAL ORDER (CHEATSHEET)**

### **Phase 1 – Foundations**

1. AST parsing
2. Depth/cost metrics
3. Feature extraction

### **Phase 2 – Data**

4. Normal generator
5. Malicious generator
6. Traffic sequence generator
7. Combine datasets

### **Phase 3 – ML**

8. Build models
9. Train model
10. Export model

### **Phase 4 – Real-time detector**

11. Rule engine
12. Score fusion
13. Detection service

### **Phase 5 – Integration**

14. Middleware
15. Logging & monitoring

---
