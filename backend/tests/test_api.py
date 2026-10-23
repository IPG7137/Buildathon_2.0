"""
PS01 Quality Inspection API Test Suite
======================================
Tests all core requirements:
- /health readiness & model loading
- Auth & RBAC (operator, reviewer, manager, admin)
- Products and Batches creation/listing
- Predict and Inspection persistence
- Evidence-Aware multi-photo sessions
- Human review (preserves original prediction)
- Corrective actions lifecycle
- Batch hold/status change authorization
- Dashboard summary & trends (DB-derived metrics)
- Grounded AI Quality Assistant
- Error handling on invalid images/formats
"""

import sys
import io
import time
import uuid
from pathlib import Path
from PIL import Image

import pytest
from fastapi.testclient import TestClient

# Ensure root is in path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.main import app
from backend.db.database import init_db, get_db, User, Product, Batch, Inspection

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Ensure DB tables and demo users are initialized and model is loaded."""
    init_db()
    from backend.ml.inference import engine as ml_engine
    ml_engine.load()


def get_token(username: str, password: str) -> str:
    """Helper to authenticate and get JWT bearer token."""
    res = client.post("/auth/token", data={"username": username, "password": password})
    assert res.status_code == 200, f"Login failed for {username}: {res.text}"
    return res.json()["access_token"]


def get_op_token() -> str:
    return get_token("operator1", "Operator@PS01Demo")


def get_rev_token() -> str:
    return get_token("reviewer1", "Reviewer@PS01Demo")


def get_mgr_token() -> str:
    return get_token("manager1", "Manager@PS01Demo")


def get_admin_token() -> str:
    return get_token("admin", "Admin@PS01Demo")


def create_dummy_image(color=(128, 128, 128), size=(224, 224)) -> bytes:
    """Generate a clean test JPEG in memory."""
    buf = io.BytesIO()
    img = Image.new('RGB', size, color=color)
    img.save(buf, format='JPEG')
    return buf.getvalue()


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_health():
    """Test /health returns ok and model status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_ready" in data
    assert "model_version" in data


def test_auth_and_roles():
    """Test logins for each role and verify JWT contents."""
    roles = [
        ("operator1", "Operator@PS01Demo", "operator"),
        ("reviewer1", "Reviewer@PS01Demo", "reviewer"),
        ("manager1", "Manager@PS01Demo", "manager"),
        ("admin", "Admin@PS01Demo", "admin"),
    ]
    for username, password, expected_role in roles:
        token = get_token(username, password)
        res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        user = res.json()
        assert user["username"] == username
        assert user["role"] == expected_role


def test_rbac_protection():
    """Verify operators cannot access manager endpoints."""
    token = get_op_token()
    res = client.get("/dashboard/summary", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code in (401, 403), "Operator should not access dashboard summary"


def test_products_and_batches():
    """Test listing products and creating a batch."""
    token = get_mgr_token()
    
    # List products
    res = client.get("/products", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    products = res.json()
    assert len(products) > 0
    prod_id = products[0]["id"]
    
    # Create batch
    batch_code = f"TEST-BATCH-{int(time.time())}"
    res = client.post("/batches", json={
        "batch_code": batch_code,
        "product_id": prod_id,
        "quantity": 500,
        "notes": "Automated test batch"
    }, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code in (200, 201)
    batch = res.json()
    assert batch["batch_code"] == batch_code
    assert batch["quality_status"] == "pending"


def test_inspection_prediction_and_persistence():
    """Test posting an inspection with an image, verifying inference & persistence."""
    token = get_op_token()
    
    # Get batches list
    res = client.get("/batches", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    batches_data = res.json()
    items = batches_data.get("items", batches_data)
    assert len(items) > 0
    batch = items[0]
    
    img_bytes = create_dummy_image(color=(100, 100, 100))
    
    # Submit inspection
    res = client.post(
        "/inspections",
        data={
            "batch_id": batch["id"],
            "capture_mode": "upload",
            "serial_number": "SN-TEST-001",
        },
        files={"file": ("test_sample.jpg", img_bytes, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code in (200, 201), f"Inspection creation failed: {res.text}"
    insp = res.json()
    assert "id" in insp
    assert insp["prediction"] in ("GOOD", "DEFECT", "UNCERTAIN")
    assert "model_score" in insp
    assert "model_version" in insp
    assert insp["inspection_status"] in ("completed", "under_review", "failed")
    assert insp["recommended_action"] is not None


def test_evidence_aware_multi_photo_session():
    """
    Test Evidence-Aware Inspection Innovation:
    Associating multiple photographs of the same physical item with one session.
    """
    token = get_op_token()
    batches_data = client.get("/batches", headers={"Authorization": f"Bearer {token}"}).json()
    batch = batches_data["items"][0]
    
    session_id = f"SESSION-EA-{int(time.time())}"
    
    # Photo 1: Top view
    img1 = create_dummy_image(color=(80, 80, 80))
    res1 = client.post(
        "/inspections",
        data={
            "batch_id": batch["id"],
            "capture_mode": "camera",
            "serial_number": "ITEM-MULTI-01",
            "session_id": session_id,
            "photo_index": 1,
            "is_primary": "true",
        },
        files={"file": ("top_view.jpg", img1, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res1.status_code in (200, 201)
    insp1 = res1.json()
    assert insp1["session_id"] == session_id
    
    # Photo 2: Side view (same session, second evidence frame)
    img2 = create_dummy_image(color=(90, 90, 90))
    res2 = client.post(
        "/inspections",
        data={
            "batch_id": batch["id"],
            "capture_mode": "camera",
            "serial_number": "ITEM-MULTI-01",
            "session_id": session_id,
            "photo_index": 2,
            "is_primary": "false",
        },
        files={"file": ("side_view.jpg", img2, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res2.status_code in (200, 201)
    insp2 = res2.json()
    assert insp2["session_id"] == session_id
    assert insp2["id"] != insp1["id"]


def test_human_review_preserves_model_prediction():
    """
    Test Human Review Queue:
    Reviewer submits a confirmed outcome.
    Must create a separate review record and preserve original model prediction.
    """
    token_op = get_op_token()
    token_rev = get_rev_token()
    batches_data = client.get("/batches", headers={"Authorization": f"Bearer {token_op}"}).json()
    batch = batches_data["items"][0]
    
    # Create inspection
    img = create_dummy_image()
    insp_res = client.post(
        "/inspections",
        data={"batch_id": batch["id"]},
        files={"file": ("test.jpg", img, "image/jpeg")},
        headers={"Authorization": f"Bearer {token_op}"}
    )
    assert insp_res.status_code in (200, 201)
    insp = insp_res.json()
    insp_id = insp["id"]
    original_pred = insp["prediction"]
    
    # Reviewer confirms defect or good
    confirmed = "GOOD" if original_pred == "DEFECT" else "DEFECT"
    res = client.post(
        f"/inspections/{insp_id}/review",
        json={
            "confirmed_class": confirmed,
            "notes": "Verified under microscope",
            "defect_category": "Scratches" if confirmed == "DEFECT" else None,
        },
        headers={"Authorization": f"Bearer {token_rev}"}
    )
    assert res.status_code in (200, 201)
    review_data = res.json()
    assert review_data["confirmed_class"] == confirmed
    
    # Verify original prediction is preserved on the inspection record
    fetched = client.get(f"/inspections/{insp_id}", headers={"Authorization": f"Bearer {token_rev}"}).json()
    assert fetched["prediction"] == original_pred, "Original model prediction MUST be preserved"
    assert len(fetched["reviews"]) > 0
    assert fetched["reviews"][0]["confirmed_class"] == confirmed
    assert fetched["inspection_status"] == "reviewed"


def test_corrective_action_and_reinspection():
    """Test creating a corrective action, updating status, and linking reinspection."""
    token = get_rev_token()
    batches_data = client.get("/batches", headers={"Authorization": f"Bearer {token}"}).json()
    batch = batches_data["items"][0]
    
    # Create an inspection
    img = create_dummy_image()
    insp = client.post(
        "/inspections",
        data={"batch_id": batch["id"]},
        files={"file": ("test.jpg", img, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"}
    ).json()
    
    # Create corrective action
    res = client.post(
        "/corrective-actions",
        json={
            "title": "Mold recalibration needed",
            "inspection_id": insp["id"],
            "batch_id": batch["id"],
            "description": "Inspect casting mold alignment and calibrate temperature",
            "priority": "high",
            "assigned_to": "reviewer1",
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code in (200, 201)
    ca = res.json()
    ca_id = ca["id"]
    assert ca["status"] == "open"
    
    # Update corrective action to in_progress
    res = client.patch(
        f"/corrective-actions/{ca_id}",
        json={"status": "in_progress", "completion_notes": "Tooling recalibrated"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    assert res.json()["status"] == "in_progress"


def test_batch_hold_and_release_workflow():
    """Test batch hold and release with authorization."""
    token = get_mgr_token()
    batches_data = client.get("/batches", headers={"Authorization": f"Bearer {token}"}).json()
    batch_id = batches_data["items"][0]["id"]
    
    # Hold batch via PATCH /batches/{id}/status
    res = client.patch(
        f"/batches/{batch_id}/status",
        json={"quality_status": "on_hold", "reason": "Excessive defect cluster detected on line 2"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    assert res.json()["quality_status"] == "on_hold"
    
    # Return batch to in_inspection
    res = client.patch(
        f"/batches/{batch_id}/status",
        json={"quality_status": "in_inspection", "reason": "Secondary check initiated"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    assert res.json()["quality_status"] == "in_inspection"


def test_dashboard_summary_and_trends():
    """Test dashboard summary returns authoritative database metrics."""
    token = get_mgr_token()
    
    # Summary
    res = client.get("/dashboard/summary", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    summary = res.json()
    assert "total_inspections" in summary
    assert "good_predictions" in summary
    assert "defect_suspected" in summary
    assert "defect_rate_pct" in summary
    assert "confirmed_defect" in summary
    assert "confirmed_good" in summary
    
    # Trends
    res = client.get("/dashboard/trends?days=7", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    trends = res.json()
    assert "daily_trends" in trends
    assert isinstance(trends["daily_trends"], list)


def test_grounded_ai_assistant():
    """Test AI assistant answers from database without fabrication."""
    token = get_mgr_token()
    
    # Query today's inspections
    res = client.get("/assistant/query?q=How many items were inspected today?", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] is not None
    assert "inspected" in data["answer"].lower() or "today" in data["answer"].lower()
    assert data["data_source"] is not None
    assert data["caveat"] is not None


def test_invalid_image_handling():
    """Test system rejects corrupt or invalid files gracefully."""
    token = get_op_token()
    batches_data = client.get("/batches", headers={"Authorization": f"Bearer {token}"}).json()
    batch = batches_data["items"][0]
    
    corrupt_data = b"NOT_A_VALID_IMAGE_FILE_DATA_CORRUPT"
    res = client.post(
        "/inspections",
        data={"batch_id": batch["id"]},
        files={"file": ("corrupted.jpg", corrupt_data, "image/jpeg")},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code in (400, 422), "Corrupt image should be rejected with 400 or 422"


def test_smart_review_priority_queue():
    """Test smart review priority queue calculates explainable risk scores."""
    token = get_rev_token()
    res = client.get("/inspections/review-queue", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    queue = data["items"]
    assert isinstance(queue, list)
    if queue:
        item = queue[0]
        assert "priority_score" in item
        assert 0 <= item["priority_score"] <= 100
        assert "priority_factors" in item
        assert isinstance(item["priority_factors"], list)


def test_batch_hold_and_release_endpoints():
    """Test authorized batch hold and release with mandatory reason logging."""
    token_mgr = get_mgr_token()
    # Create fresh batch to avoid pending review conflict
    prods = client.get("/products", headers={"Authorization": f"Bearer {token_mgr}"}).json()
    unique_code = f"BATCH-TEST-{uuid.uuid4().hex[:6].upper()}"
    b_res = client.post(
        "/batches",
        json={"product_id": prods[0]["id"], "batch_code": unique_code, "quantity": 50},
        headers={"Authorization": f"Bearer {token_mgr}"}
    )
    assert b_res.status_code in (200, 201)
    batch_id = b_res.json()["id"]

    # Quarantine on hold
    res_hold = client.post(
        f"/batches/{batch_id}/hold",
        json={"reason": "Suspected cluster of micro-cracks on Line 1"},
        headers={"Authorization": f"Bearer {token_mgr}"}
    )
    assert res_hold.status_code == 200
    assert res_hold.json()["quality_status"] == "on_hold"
    assert res_hold.json()["hold_reason"] == "Suspected cluster of micro-cracks on Line 1"

    # Authorized release
    res_rel = client.post(
        f"/batches/{batch_id}/release",
        json={"reason": "Secondary ultrasonic inspection verified zero sub-surface fissures"},
        headers={"Authorization": f"Bearer {token_mgr}"}
    )
    assert res_rel.status_code == 200
    assert res_rel.json()["quality_status"] == "approved"


def test_capa_repair_evidence_and_verification():
    """Test CAPA workflow: upload repair evidence photo, moves to awaiting reinspection, and passes verification."""
    token_rev = get_rev_token()
    token_op = get_op_token()
    
    # Get or create an inspection
    batches = client.get("/batches", headers={"Authorization": f"Bearer {token_op}"}).json()["items"]
    img = create_dummy_image()
    insp = client.post(
        "/inspections",
        data={"batch_id": batches[0]["id"]},
        files={"file": ("defect.jpg", img, "image/jpeg")},
        headers={"Authorization": f"Bearer {token_op}"}
    ).json()

    # Create CAPA
    ca_res = client.post(
        "/corrective-actions",
        json={"inspection_id": insp["id"], "title": "Surface grinding repair", "priority": "high", "root_cause": "Roller friction"},
        headers={"Authorization": f"Bearer {token_rev}"}
    )
    assert ca_res.status_code in (200, 201)
    ca_id = ca_res.json()["id"]

    # Upload after-repair evidence
    repair_img = create_dummy_image(color=(180, 180, 180))
    res_ev = client.post(
        f"/corrective-actions/{ca_id}/evidence",
        data={"notes": "Grinded surface smooth to 0.1 Ra and buffed"},
        files={"file": ("after_repair.jpg", repair_img, "image/jpeg")},
        headers={"Authorization": f"Bearer {token_rev}"}
    )
    assert res_ev.status_code == 200
    assert res_ev.json()["status"] == "awaiting_reinspection"
    assert res_ev.json()["after_image_path"] is not None

    # Reviewer verifies and passes reinspection
    res_ver = client.post(
        f"/corrective-actions/{ca_id}/verify",
        json={"decision": "passed", "notes": "Optical surface roughness within tolerance"},
        headers={"Authorization": f"Bearer {token_rev}"}
    )
    assert res_ver.status_code == 200
    assert res_ver.json()["status"] == "resolved"
    assert res_ver.json()["reinspection_status"] == "passed"


def test_quality_summary_report():
    """Test executive quality report compiles valid compliance metrics."""
    token = get_mgr_token()
    res = client.get("/reports/quality-summary", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    report = res.json()
    assert "executive_summary" in report
    assert "total_inspections" in report["executive_summary"]
    assert "defect_rate_pct" in report["executive_summary"]
    assert "system_status" in report
    assert report["system_status"]["model_architecture"] == "MobileNetV3-Small"


def test_audit_events_ledger():
    """Test audit events ledger records actions and supports query filtering."""
    token = get_mgr_token()
    res = client.get("/audit/events", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    events = res.json()
    assert "items" in events
    assert isinstance(events["items"], list)
    assert len(events["items"]) > 0
    evt = events["items"][0]
    assert "action" in evt
    assert "actor_name" in evt
    assert "timestamp" in evt


if __name__ == "__main__":
    pytest.main(["-v", __file__])


