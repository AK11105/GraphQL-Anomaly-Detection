# **PHASE 1 TIMELINE — FOUNDATIONS (AST → METRICS → FEATURES)**

### **1. Implement AST extraction**

* Build `ast_extractor.js`.
* Ensure any valid GraphQL query parses to AST.
* Add schema-aware node walking helpers.

**Why:**
Everything else depends on parsing.
Without AST, no depth, cost, feature extraction, or generation works.

---

### **2. Implement depth metrics**

* Max depth
* Avg depth
* Breadth (children per node)
* Total node count

**Why:**
These are core features AND used to generate malicious queries later.

---

### **3. Implement cost / complexity metrics**

* Wrap `graphql-query-complexity`
* Compute cost score & cost breakdown

**Why:**
Your anomaly model needs both depth AND cost signals.

---

### **4. Implement metadata extractors**

* alias count
* fragment count
* token length
* entropy

**Why:**
These additional metrics drastically improve ML accuracy.

---

### **5. Implement combined feature extractor**

* Merge AST features, depth metrics, cost metrics, entropy
* Produce final feature vector structure
* Store schema in config file

**Why:**
The ML pipeline and data generation depend on a fixed feature schema.

---

### **PHASE 1 EXIT CONDITION**

You can take ANY query → extract AST → compute metrics → produce ML feature vector.

---