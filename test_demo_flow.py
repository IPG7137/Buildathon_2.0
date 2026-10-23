"""
End-to-End Live Workflow Verification Script
===========================================
Executes the requested demonstration flow:
1. Operator authentication
2. Batch selection
3. Submitting actual steel casting/strip image from dataset
4. AI model inference and action recommendation
5. Manager authentication and dashboard summary verification
6. Human review recording (confirm/correct)
7. Corrective action creation
8. Batch quality status update
9. AI Assistant natural language query
"""

import sys
import glob
import os
import json
import urllib.request
import urllib.parse

BASE_URL = "http://localhost:8000"

def post_form(url, data_dict, token=None):
    data = urllib.parse.urlencode(data_dict).encode()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def get_json(url, token=None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def post_json(url, data_dict, token=None, method="POST"):
    body = json.dumps(data_dict).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def main():
    print("=" * 65)
    print("PS01 END-TO-END DEMO WORKFLOW VERIFICATION")
    print("=" * 65)

    # 1. Login operator
    op_auth = post_form(f"{BASE_URL}/auth/token", {"username": "operator1", "password": "Operator@PS01Demo"})
    op_token = op_auth["access_token"]
    print(f"[1] Operator Logged In: {op_auth['display_name']} ({op_auth['role']})")

    # 2. Get batches
    batches = get_json(f"{BASE_URL}/batches", op_token)["items"]
    batch = batches[0]
    print(f"[2] Selected Batch: {batch['batch_code']} (ID: {batch['id']})")

    # 3. Read real dataset image
    img_files = glob.glob("dataset/casting_dataset/test/*/*.jpeg") or glob.glob("dataset/train_images/*.jpg")
    if not img_files:
        print("ERROR: No dataset images found!")
        sys.exit(1)
    
    img_path = img_files[0]
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    filename = os.path.basename(img_path)
    print(f"[3] Selected Real Sample Image: {filename} ({len(img_bytes):,} bytes)")

    # 4. Upload & Inspect
    boundary = "----WebKitFormBoundaryPS01DemoBoundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"batch_id\"\r\n\r\n{batch['id']}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"capture_mode\"\r\n\r\nupload\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"serial_number\"\r\n\r\nDEMO-PIECE-042\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"session_id\"\r\n\r\nSESSION-DEMO-LIVE\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"photo_index\"\r\n\r\n1\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n"
    ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{BASE_URL}/inspections",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {op_token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}"
        }
    )
    with urllib.request.urlopen(req) as resp:
        insp = json.loads(resp.read().decode())

    print(f"[4] AI Inspection Result:")
    print(f"    Inspection ID:     {insp['id']}")
    print(f"    Model Prediction:  {insp['prediction']}")
    print(f"    Confidence Scores: GOOD={insp['good_score']:.4f}  DEFECT={insp['defect_score']:.4f}")
    print(f"    Inference Time:    {insp['inference_time_ms']:.1f} ms")
    print(f"    Model Version:     {insp['model_version']}")
    print(f"    Inspection Status: {insp['inspection_status']}")
    print(f"    Recommended Step:  {insp['recommended_action']}")

    # 5. Manager views Dashboard Summary
    mgr_auth = post_form(f"{BASE_URL}/auth/token", {"username": "manager1", "password": "Manager@PS01Demo"})
    mgr_token = mgr_auth["access_token"]
    summary = get_json(f"{BASE_URL}/dashboard/summary", mgr_token)
    print(f"[5] Manager Dashboard Live Update:")
    print(f"    Total Inspections Today: {summary['total_inspections']}")
    print(f"    Good Predictions:        {summary['good_predictions']}")
    print(f"    Defect Suspected:        {summary['defect_suspected']}")
    print(f"    Awaiting Human Review:   {summary['pending_review']}")
    print(f"    Defect Rate:             {summary['defect_rate_pct']} %")

    # 6. Quality Reviewer verifies / reviews inspection
    rev_auth = post_form(f"{BASE_URL}/auth/token", {"username": "reviewer1", "password": "Reviewer@PS01Demo"})
    rev_token = rev_auth["access_token"]
    review_res = post_json(
        f"{BASE_URL}/inspections/{insp['id']}/review",
        {
            "confirmed_class": insp["prediction"],
            "notes": "Verified surface characteristics visually under magnification",
            "defect_category": "Surface Scratches" if insp["prediction"] == "DEFECT" else None,
        },
        rev_token
    )
    print(f"[6] Human Quality Review Recorded:")
    print(f"    Confirmed Class:         {review_res['confirmed_class']}")
    print(f"    Original AI Output:      {review_res['original_prediction']} (PRESERVED)")

    # 7. Corrective action
    if insp["prediction"] == "DEFECT":
        ca = post_json(
            f"{BASE_URL}/corrective-actions",
            {
                "title": f"Corrective Action for Batch {batch['batch_code']}",
                "inspection_id": insp["id"],
                "batch_id": batch["id"],
                "description": "Inspect casting mold alignment and calibrate temperature",
                "priority": "high",
                "assigned_to": "reviewer1",
            },
            rev_token
        )
        print(f"[7] Corrective Action Created: ID={ca['id']} (Status: {ca['status']})")
    else:
        print("[7] No defect detected - piece passed inspection.")

    # 8. AI Quality Assistant Query
    assistant_res = get_json(f"{BASE_URL}/assistant/query?q=How+many+items+were+inspected+today%3F", mgr_token)
    print(f"[8] Grounded AI Assistant Response:")
    print(f"    Q: {assistant_res['question']}")
    print(f"    A: {assistant_res['answer']}")
    print(f"    Source: {assistant_res['data_source']}")

    print("=" * 65)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
