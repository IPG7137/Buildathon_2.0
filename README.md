# QualityGuard AI
## AI-Powered Industrial Product Quality Inspection & Defect Tracking System
### Buildathon 2.0 — Third-Year Track

---

## 1. Executive Summary & Problem Statement

Modern manufacturing facilities face acute quality-control challenges: human visual fatigue causes defect escape rates of 15–30%, manual record-keeping leads to fragmented audit trails, and ad-hoc rework lacks verifiable repair documentation.

**QualityGuard AI** is an industrial-grade quality management and vision inspection platform designed for high-throughput discrete manufacturing (cold-rolled steel strips and precision metal castings). The system integrates:
1. **Edge/Mobile Live Camera Studio**: Browser-native camera stream with rear-facing hardware selection, interactive framing reticle, and a 2-second capture countdown delay for motion blur suppression.
2. **Real Deep Learning Vision Inference**: PyTorch MobileNetV3-Small neural network trained on genuine industrial defect datasets, achieving **84.07% accuracy** on a 452-image held-out test set.
3. **Smart Review Priority Queue (Innovation)**: An explainable 0–100 risk scoring algorithm that prioritizes human QC reviews based on model uncertainty, defect probability, batch quarantine status, and defect recurrence patterns.
4. **Authoritative Human-in-the-Loop Sign-off**: Immutability architecture where AI predictions are strictly advisory; authorized QC reviewers confirm or correct findings without overwriting the original model outputs.
5. **Traceable CAPA & Before/After Repair Evidence**: End-to-end corrective action lifecycle tracking root causes, before/after photographic proof, and reinspection approval.
6. **Grounded QualityGuard Copilot**: A generative AI assistant that answers questions strictly using verified relational database records and provides clickable entity citations (`[INSP-xxxx]`, `[BATCH-xxxx]`, `[CAPA-xxxx]`).
7. **Immutable Audit Ledger**: Comprehensive change tracking logging actor, action, previous state, new state, and mandatory human justification.

---

## 2. System Architecture

```
                                  +------------------------------------------------------+
                                  |              CLIENT APPLICATIONS                     |
                                  |                                                      |
                                  |  +--------------------+    +----------------------+  |
                                  |  | Manager Dashboard  |    | Mobile Inspection App|  |
                                  |  | (Chart.js / Studio)|    | (PWA / getUserMedia) |  |
                                  +--+---------+----------+----+----------+-----------+--+
                                               |                          |
                                    REST / JSON / Multipart HTTP Requests |
                                               |                          |
                                  +------------v--------------------------v--------------+
                                  |               FASTAPI BACKEND CORE                   |
                                  |                                                      |
                                  |  [ JWT Auth & RBAC ]    [ Input Validation ]         |
                                  |  - operator / reviewer  - File size / Magic bytes    |
                                  |  - manager / admin      - Path traversal guard       |
                                  |                                                      |
                                  |  +------------------+   +-------------------------+  |
                                  |  | ML Inference Svc |   | Smart Priority Queue    |  |
                                  |  | (PyTorch Mobile- |   | (Explainable Risk 0-100)|  |
                                  |  |  NetV3-Small)    |   |                         |  |
                                  |  +--------+---------+   +------------+------------+  |
                                  |           |                          |               |
                                  |  +--------v---------+   +------------v------------+  |
                                  |  | Grounded Copilot |   | Executive Quality Report|  |
                                  |  | (Entity Citations|   | (Printable Compliance)  |  |
                                  +--+--------+---------+---+------------+------------+--+
                                              |                          |
                                              +------------+-------------+
                                                           |
                                  +------------------------v-----------------------------+
                                  |                 PERSISTENCE LAYER                    |
                                  |                                                      |
                                  |  +--------------------+    +----------------------+  |
                                  |  | SQLite / Postgres  |    | Evidence Storage     |  |
                                  |  | (WAL Mode / Relat- |    | (backend/storage/    |  |
                                  |  |  ional Integrity)  |    |  images/)            |  |
                                  +------------------------------------------------------+
```

---

## 3. Machine Learning Model & Dataset Inspection

### Datasets Evaluated
1. **Severstal Steel Defect Detection**:
   - Sampled and verified: **3,000 industrial surface images** (1,611 Defect / 1,389 Good).
   - Format: Single-channel and 3-channel cold-rolled steel surface scans.
   - Defect classes: Pitting, inclusion clusters, surface scratches, roll seams.
2. **Casting Defect Dataset (dataset2)**:
   - 7,348 metal casting photos (submersible pump impellers).
   - Used for multi-category testing and cross-domain validation.

### Actual Held-Out Model Evaluation (Severstal Steel Model)
- **Model Architecture**: MobileNetV3-Small (`torchvision.models.mobilenet_v3_small`)
- **Active Weights Path**: `backend/ml/checkpoints/model_v1.pth` (Also archived at `models/model_v1.pth`)
- **Held-Out Test Set**: **452 physical product images** (243 Defect / 209 Good)
- **Overall Accuracy**: **84.07%** (380 / 452 correct)

| Metric | GOOD (Normal Surface) | DEFECT (Surface Flaw) | Macro Average |
|---|---|---|---|
| **Precision** | **0.810** | **0.870** | **0.840** |
| **Recall** | **0.857** | **0.827** | **0.842** |
| **F1-Score** | **0.833** | **0.848** | **0.841** |
| **Test Support** | 209 images | 243 images | 452 images |

#### Quality-Critical Safety Metrics:
- **False Acceptance Rate (FAR)** (Defect missed as Good): **17.28%** (42 / 243)
- **False Rejection Rate (FRR)** (Good flagged as Defect): **14.35%** (30 / 209)
- **Average Inference Latency**: **18–26 ms** per frame on standard CPU.

---

## 4. Key Feature Walkthrough

### A. Live Camera Inspection Studio (Priority Hackathon Feature)
- **Mode A — Live Camera**:
  1. Access via dashboard sidebar (*Live Inspection*) or mobile app.
  2. Requests camera stream (`navigator.mediaDevices.getUserMedia`) with `facingMode: "environment"` (rear-facing camera on mobile).
  3. Displays active video stream with industrial targeting reticle and corner alignment brackets.
  4. User clicks **Capture Frame (2s Delay)**: A full-screen visual countdown overlay (`2... 1... CAPTURING!`) provides a 2-second delay to stabilize the physical product.
  5. Captures sharp still frame on canvas, freezes preview, and offers **Retake** vs **Run AI Inspection**.
  6. Submits to `POST /inspections` for real-time PyTorch inference.
- **Mode B — Photo Upload**: Drag-and-drop or file picker with format (JPG/PNG/WebP) and size (≤ 20 MB) validation.
- **Mode C — Curated Dataset Samples**: One-click test cards allowing judges to test real Severstal Steel and Casting images immediately.

### B. Smart Review Priority Queue (Innovation Feature)
Prioritizes inspections for human QC sign-off using an explainable 0–100 risk score:
- **Model Uncertainty (+35 pts)**: When defect score falls within `[0.35, 0.65]`.
- **Defect Probability (+25 pts)**: Proportional to raw model defect confidence.
- **Batch Risk Status (+20 pts)**: If the associated production batch is currently quarantined on hold.
- **Defect Recurrence (+15 pts)**: If the product has historical recurring defects on that production line.
- **Queue Wait Time (+15 pts)**: Added if inspection has been awaiting review for > 2 hours.

### C. Traceable Reinspection & Before/After Evidence
1. Inspection flags defect $\rightarrow$ Authoritative reviewer confirms outcome.
2. Corrective Action (CAPA) is created with title, priority, and root cause hypothesis.
3. Rework team repairs component and uploads **After-Repair Evidence Image** (`POST /corrective-actions/{id}/evidence`).
4. Status transitions to `awaiting_reinspection`.
5. Reviewer opens **Side-by-Side Before/After Comparison** modal and verifies repair:
   - **Passed**: Resolves CAPA and records closure in audit ledger.
   - **Failed**: Reopens CAPA for secondary remediation.

### D. Grounded QualityGuard Copilot
- Connected strictly to live SQLite database records.
- Queries answer questions regarding batch defect summaries, overdue actions, and recurrence patterns.
- Formats answers with clickable citation tags: `[INSP-xxxx]`, `[BATCH-xxxx]`, `[CAPA-xxxx]`.

---

## 5. Role-Based Access Control (RBAC) Matrix

| Endpoint / Action | Operator | QC Reviewer | Quality Manager | System Admin |
|---|:---:|:---:|:---:|:---:|
| **Run Camera / Upload Inspection** (`POST /inspections`) | ✅ | ✅ | ✅ | ✅ |
| **View Inspections & History** (`GET /inspections`) | ✅ | ✅ | ✅ | ✅ |
| **View Smart Priority Queue** (`GET /inspections/review-queue`) | ❌ | ✅ | ✅ | ✅ |
| **Submit Authoritative QC Review** (`POST /inspections/{id}/review`) | ❌ | ✅ | ✅ | ✅ |
| **Create CAPA Ticket** (`POST /corrective-actions`) | ❌ | ✅ | ✅ | ✅ |
| **Upload After-Repair Evidence** (`POST /corrective-actions/{id}/evidence`) | ❌ | ✅ | ✅ | ✅ |
| **Verify Reinspection Decision** (`POST /corrective-actions/{id}/verify`) | ❌ | ✅ | ✅ | ✅ |
| **Quarantine Batch On Hold** (`POST /batches/{id}/hold`) | ❌ | ❌ | ✅ | ✅ |
| **Release Batch to Dispatch** (`POST /batches/{id}/release`) | ❌ | ❌ | ✅ | ✅ |
| **Register Products & Batches** (`POST /products`, `POST /batches`) | ❌ | ❌ | ✅ | ✅ |
| **View Audit Trail Ledger** (`GET /audit/events`) | ❌ | ✅ | ✅ | ✅ |

---

## 6. Demo Accounts (Pre-Seeded)

| Role | Username | Password | Key Privileges |
|---|---|---|---|
| **Quality Manager** | `manager1` | `Manager@PS01Demo` | Batch hold/release disposition, analytics, executive reports |
| **QC Reviewer** | `reviewer1` | `Reviewer@PS01Demo` | Priority review queue, defect verification, CAPA reinspection |
| **Inspector / Operator (A)** | `operator1` | `Operator@PS01Demo` | Live camera studio, photo uploads, session grouping |
| **Inspector / Operator (B)** | `operator2` | `Operator2@PS01Demo` | Line 2 floor operator inspections |
| **System Admin** | `admin` | `Admin@PS01Demo` | Complete system administration and configuration |

---

## 7. Setup & Execution Instructions

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9)
- PyTorch 2.0+ & Torchvision
- Modern browser (Chrome / Edge / Firefox / Safari)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-seed Demo Database
The repository includes a populated SQLite database at `backend/db/ps01.db`. To reset to clean demonstration records:
```bash
python backend/db/database.py
```

### Step 3: Start QualityGuard AI
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Access the Applications
- **Industrial Manager Dashboard**: [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)
- **Mobile Inspection PWA**: [http://localhost:8000/mobile/](http://localhost:8000/mobile/)
- **Interactive OpenAPI Specification**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check & Model Diagnostic**: [http://localhost:8000/health](http://localhost:8000/health)

### Step 5: Run Automated Test Suite
```bash
pytest backend/tests/test_api.py -v
```

---

## 8. Verified End-to-End Test Suite

The automated test suite in `backend/tests/test_api.py` covers 17 test suites:
- `test_health`: API readiness, model weights verification.
- `test_auth_and_roles`: Token issue, role enforcement.
- `test_rbac_protection`: Blocks unauthorized actions.
- `test_products_and_batches`: Product catalog and batch CRUD.
- `test_inspection_prediction_and_persistence`: Live inference and SQLite storage.
- `test_evidence_aware_multi_photo_session`: Multi-frame inspection grouping.
- `test_human_review_preserves_model_prediction`: Review audit separation.
- `test_corrective_action_and_reinspection`: Full CAPA lifecycle.
- `test_batch_hold_and_release_workflow`: Human disposition authorization.
- `test_dashboard_summary_and_trends`: Metric calculations.
- `test_grounded_ai_assistant`: Copilot factual database queries.
- `test_invalid_image_handling`: Corrupt/unsupported file rejection.
- `test_smart_review_priority_queue`: Explainable 0-100 risk scoring.
- `test_batch_hold_and_release_endpoints`: Hold/Release justification logging.
- `test_capa_repair_evidence_and_verification`: Post-repair evidence and verification.
- `test_quality_summary_report`: Executive compliance compilation.
- `test_audit_events_ledger`: Immutable audit trail.

---

## 9. Hardware & Dataset Compatibility Notes
- **Inference Hardware**: PyTorch CPU mode enabled with AVX2 acceleration. Latency is 18–26 ms per frame on Intel/AMD x86_64, eliminating GPU dependency for buildathon evaluation.
- **Model Checkpoint**: Genuine PyTorch weights saved in `backend/ml/checkpoints/model_v1.pth` (SHA-256 verified) and `models/model_v1.pth`.
