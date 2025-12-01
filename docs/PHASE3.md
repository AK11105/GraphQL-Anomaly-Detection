# **PHASE 3 TIMELINE — ML PIPELINE (MODELS → TRAINING → EXPORT)**

### **1. Implement model architectures**

* LSTM anomaly detector
* Optional Transformer version
* Ensemble head for static + temporal features

**Why:**
All training depends on having the architecture first.

---

### **2. Build model training pipeline**

* Dataloader for sequences
* Loss function
* Optimizer
* Metrics: AUC, precision, recall
* Logging

**Why:**
Makes training repeatable and debuggable.

---

### **3. Run experiments**

* Train LSTM on base dataset
* Evaluate on malicious variants
* Tune hyperparameters
* Document performance

**Why:**
Ensures you have a production-ready model.

---

### **4. Export final model formats**

* TorchScript
* ONNX

**Why:**
Required for real-time serving in Phase 4.

---

### **PHASE 3 EXIT CONDITION**

You have a final trained model + exported inference artifact.

---