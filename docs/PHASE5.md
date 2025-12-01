# **PHASE 5 TIMELINE — INTEGRATION INTO GRAPHQL SERVER**

### **1. Build server middleware**

* `express_middleware.js`
* Capture raw queries
* Extract features
* Send to realtime detector
* Act on result

**Why:**
This connects your detection system to real workloads.

---

### **2. Add structured logging**

* Query metadata
* action taken
* scores
* latency

**Why:**
Needed for retraining, debugging, audits.

---

### **3. Add monitoring**

* Prometheus metrics
* Grafana dashboards

**Why:**
Essential for production deployment.

---

### **4. Dockerize components**

* ingestion
* ml-serving
* realtime-detector
* integration layer

**Why:**
Allows consistent deployment across environments.

---

### **5. K8s manifests**

* deployments
* services
* configmaps
* autoscaling policies

**Why:**
Enables real production rollout.

---

### **PHASE 5 EXIT CONDITION**

Your GraphQL server is protected by a fully integrated anomaly detection system.

---