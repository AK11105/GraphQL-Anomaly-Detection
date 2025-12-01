# **PHASE 4 TIMELINE — REAL-TIME DETECTOR (RULES + ML + ACTIONS)**

### **1. Implement rule engine**

* Integrate `graphql-cop`
* Compute rule severity score
* Flag structural anomalies

**Why:**
Rule engine provides zero-latency baseline checks.

---

### **2. Implement ML inference service**

* Load TorchScript/ONNX model
* Run inference on incoming feature vectors
* Return anomaly probability

**Why:**
This provides the dynamic part of risk scoring.

---

### **3. Implement score fusion logic**

* Combine: rule score + ML score
* Weighting controlled via config
* Produce unified risk score

**Why:**
Hybrid scoring dramatically improves accuracy.

---

### **4. Implement action handlers**

* allow
* throttle
* block

Each reacts based on threshold tiers.

**Why:**
You need real decisions, not just scores.

---

### **5. Build main real-time detector service**

* Connect ingestion → scoring → action
* Add logging, monitoring hooks
* Ensure low-latency execution

**Why:**
This is the online runtime engine.

---

### **PHASE 4 EXIT CONDITION**

You can pass a query into the system and receive allow/throttle/block in real time.

---