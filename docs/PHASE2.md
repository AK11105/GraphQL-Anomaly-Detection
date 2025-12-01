# **PHASE 2 TIMELINE — DATA PIPELINE (NORMAL + MALICIOUS + SEQUENCES)**

### **1. Normal query generator**

* Build `generate_normal.py`
* Use schema introspection from `graphql-js`
* Generate valid, realistic queries with depth 1–4
* Add argument sampling

**Why:**
You define the "normal manifold" before building anomalies.

---

### **2. Malicious query generator**

* Build `generate_malicious.py`
* Deep recursion patterns
* Fragment recursion
* Alias bombs
* Wide branching
* High entropy tokens

**Why:**
You now contrast against the normal queries.

---

### **3. Validate both normal & malicious using Phase 1 extractors**

* Pass each generated query through:

  * AST parser
  * depth calculator
  * cost calculator
  * feature extractor

**Why:**
Ensures dataset consistency and correctness.

---

### **4. Generate time-series sequences**

* Build `generate_sequences.py`
* Mix normal + malicious patterns
* Generate windows of 10–50 queries
* Add label metadata

**Why:**
Time-series models require sequential behavior, not isolated queries.

---

### **5. Run preprocessing**

* normalize numerical features
* apply sliding windows
* generate train/val/test splits

**Why:**
Creates a clean, ML-ready dataset.

---

### **PHASE 2 EXIT CONDITION**

You have labeled normal/malicious datasets + sequential windows ready for ML training.

---