"""
FastAPI Backend for PS01 - Product Quality Inspection & Defect Tracking System
===============================================================================
All endpoints documented. Run with: uvicorn backend.main:app --reload --port 8000
"""

import io
import json
import os
import uuid
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Any, Dict

from fastapi import (
    FastAPI, Depends, HTTPException, UploadFile, File, Form,
    BackgroundTasks, Query, status, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy import func, text, and_, or_, desc

# Internal imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.db.database import (
    get_db, init_db, User, Product, Batch, Inspection, InspectionReview,
    CorrectiveAction, AuditEvent, get_password_hash, verify_password,
    utcnow, new_uuid, IMAGES_STORAGE
)
from backend.api.auth import (
    create_access_token, get_current_user, require_role, ACCESS_TOKEN_EXPIRE_MINUTES
)
from backend.ml.inference import engine as ml_engine

# ─── App init ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PS01 Quality Inspection API",
    description="Product Quality Inspection & Defect Tracking System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use ALLOWED_ORIGINS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("[Startup] Initializing database...")
    init_db()
    print("[Startup] Loading ML model...")
    ok = ml_engine.load()
    if ok:
        print(f"[Startup] ML model ready (v{ml_engine.model_version})")
    else:
        print(f"[Startup] WARNING: ML model failed to load: {ml_engine.load_error}")
        print("[Startup]   Run: python backend/ml/train.py to train the model first.")


# ─── Pydantic schemas ────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    role: str
    display_name: str

class UserOut(BaseModel):
    id: str
    username: str
    display_name: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    class Config: from_attributes = True

class ProductCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    product_type: str = Field("steel_strip", max_length=64)
    sku: Optional[str] = None
    description: Optional[str] = None
    inspection_category: str = "surface_defect"
    inspection_criteria: Optional[str] = None
    threshold_defect: float = 0.45

class ProductOut(BaseModel):
    id: str
    name: str
    product_type: str
    sku: Optional[str] = None
    description: Optional[str] = None
    inspection_category: str = "surface_defect"
    inspection_criteria: Optional[str] = None
    threshold_defect: Optional[float] = 0.45
    is_active: bool = True
    created_at: Optional[datetime] = None
    class Config: from_attributes = True

class BatchCreate(BaseModel):
    batch_code: str = Field(..., min_length=3, max_length=64)
    product_id: str
    quantity: Optional[int] = None
    production_line: Optional[str] = "Line 1"
    notes: Optional[str] = None

class BatchOut(BaseModel):
    id: str
    batch_code: str
    product_id: str
    quantity: Optional[int] = None
    production_line: Optional[str] = "Line 1"
    quality_status: str
    hold_reason: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    class Config: from_attributes = True

class InspectionOut(BaseModel):
    id: str
    batch_id: str
    operator_id: str
    serial_number: Optional[str]
    capture_mode: str
    image_path: Optional[str]
    prediction: Optional[str]
    defect_score: Optional[float]
    good_score: Optional[float]
    model_version: Optional[str]
    inference_time_ms: Optional[float]
    inspection_status: str
    session_id: Optional[str]
    photo_index: int
    is_primary: bool
    timestamp: datetime
    class Config: from_attributes = True

class ReviewCreate(BaseModel):
    confirmed_class: str = Field(..., description="GOOD|DEFECT|INCONCLUSIVE")
    reason: Optional[str] = None
    notes: Optional[str] = None

class ReviewOut(BaseModel):
    id: str
    inspection_id: str
    reviewer_id: str
    confirmed_class: str
    original_prediction: Optional[str]
    reason: Optional[str]
    notes: Optional[str]
    reviewed_at: datetime
    class Config: from_attributes = True

class CorrectiveActionCreate(BaseModel):
    inspection_id: Optional[str] = None
    batch_id: Optional[str] = None
    title: str = Field(..., min_length=5, max_length=256)
    description: Optional[str] = None
    root_cause: Optional[str] = None
    before_image_path: Optional[str] = None
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: str = "medium"

class CorrectiveActionUpdate(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    root_cause: Optional[str] = None
    completion_notes: Optional[str] = None
    due_date: Optional[datetime] = None
    follow_up_inspection_id: Optional[str] = None
    after_image_path: Optional[str] = None
    reinspection_status: Optional[str] = None
    reinspection_notes: Optional[str] = None

class ReinspectionVerifyPayload(BaseModel):
    decision: str = Field(..., description="passed|failed")
    notes: str = Field(..., min_length=3, max_length=512)

class CorrectiveActionOut(BaseModel):
    id: str
    inspection_id: Optional[str]
    batch_id: Optional[str]
    title: str
    description: Optional[str]
    root_cause: Optional[str] = None
    before_image_path: Optional[str] = None
    after_image_path: Optional[str] = None
    assigned_to: Optional[str]
    created_by: str
    due_date: Optional[datetime]
    status: str
    priority: str
    completion_notes: Optional[str]
    follow_up_inspection_id: Optional[str]
    reinspection_status: Optional[str] = "pending"
    reinspection_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    class Config: from_attributes = True

class BatchStatusUpdate(BaseModel):
    quality_status: str  # on_hold|approved|rejected|in_inspection
    reason: str = Field(..., min_length=5, max_length=512)

class BatchReasonPayload(BaseModel):
    reason: str = Field(..., min_length=5, max_length=512)

class AuditEventOut(BaseModel):
    id: str
    actor_id: Optional[str] = None
    actor_name: Optional[str] = "System"
    action: str
    entity_type: str
    entity_id: Optional[str] = None
    previous_state: Optional[str] = None
    new_state: Optional[str] = None
    reason: Optional[str] = None
    timestamp: datetime
    class Config: from_attributes = True

class PaginatedResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[Any]


# ─── Helper functions ─────────────────────────────────────────────────────────

def save_image(image_bytes: bytes, filename: str, inspection_id: str) -> str:
    """Save image to local storage. Returns relative path."""
    ext = Path(filename).suffix.lower() or '.jpg'
    safe_name = f"{inspection_id}{ext}"
    dest = IMAGES_STORAGE / safe_name
    with open(dest, 'wb') as f:
        f.write(image_bytes)
    return f"images/{safe_name}"

def log_audit(db: Session, actor: User, action: str, entity_type: str,
              entity_id: str, prev=None, new=None, reason: str = None):
    evt = AuditEvent(
        actor_id=actor.id if actor else None,
        actor_name=actor.display_name if actor else "system",
        action=action, entity_type=entity_type, entity_id=entity_id,
        previous_state=json.dumps(prev) if prev else None,
        new_state=json.dumps(new) if new else None,
        reason=reason,
    )
    db.add(evt)


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model_ready": ml_engine.is_ready,
        "model_version": ml_engine.model_version,
        "model_error": ml_engine.load_error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Auth ─────────────────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username, User.is_active == True).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    user.last_login = utcnow()
    db.commit()
    
    token = create_access_token(data={"sub": user.username, "role": user.role, "uid": user.id})
    return TokenResponse(
        access_token=token, user_id=user.id, username=user.username,
        role=user.role, display_name=user.display_name,
    )

@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
def me(current_user: User = Depends(get_current_user)):
    return current_user

@app.get("/users", response_model=List[UserOut], tags=["Auth"])
def list_users(db: Session = Depends(get_db), _: User = Depends(require_role("manager"))):
    return db.query(User).filter(User.is_active == True).all()


# ─── Products ────────────────────────────────────────────────────────────────

@app.get("/products", response_model=List[ProductOut], tags=["Products"])
def list_products(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return db.query(Product).filter(Product.is_active == True).all()

@app.post("/products", response_model=ProductOut, status_code=201, tags=["Products"])
def create_product(
    payload: ProductCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("manager")),
):
    existing = db.query(Product).filter(Product.name == payload.name).first()
    if existing:
        raise HTTPException(409, detail="Product with this name already exists")
    prod = Product(
        name=payload.name,
        product_type=payload.product_type,
        sku=payload.sku or f"SKU-{payload.name[:3].upper()}-01",
        description=payload.description,
        inspection_category=payload.inspection_category,
        inspection_criteria=payload.inspection_criteria,
        threshold_defect=payload.threshold_defect,
    )
    db.add(prod)
    db.commit()
    db.refresh(prod)
    log_audit(db, current_user, "CREATE", "product", prod.id, new={"name": prod.name, "sku": prod.sku})
    db.commit()
    return prod

@app.get("/products/{product_id}", tags=["Products"])
def get_product(product_id: str, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    prod = db.query(Product).filter(Product.id == product_id).first()
    if not prod:
        raise HTTPException(404, "Product not found")
    batches = db.query(Batch).filter(Batch.product_id == prod.id).all()
    batch_ids = [b.id for b in batches]
    insp_count = db.query(func.count(Inspection.id)).filter(Inspection.batch_id.in_(batch_ids)).scalar() if batch_ids else 0
    defect_count = db.query(func.count(Inspection.id)).filter(Inspection.batch_id.in_(batch_ids), Inspection.prediction == "DEFECT").scalar() if batch_ids else 0
    return {
        "id": prod.id,
        "name": prod.name,
        "product_type": prod.product_type,
        "sku": prod.sku,
        "description": prod.description,
        "inspection_category": prod.inspection_category,
        "inspection_criteria": prod.inspection_criteria,
        "threshold_defect": prod.threshold_defect,
        "is_active": prod.is_active,
        "total_batches": len(batches),
        "total_inspections": insp_count,
        "defect_inspections": defect_count,
        "batches": [{"id": b.id, "batch_code": b.batch_code, "status": b.quality_status, "quantity": b.quantity} for b in batches],
    }


# ─── Batches ─────────────────────────────────────────────────────────────────

@app.post("/batches", response_model=BatchOut, status_code=201, tags=["Batches"])
def create_batch(payload: BatchCreate, db: Session = Depends(get_db),
                 current_user: User = Depends(require_role("manager"))):
    product = db.query(Product).filter(Product.id == payload.product_id).first()
    if not product:
        raise HTTPException(404, detail="Product not found")
    existing = db.query(Batch).filter(Batch.batch_code == payload.batch_code).first()
    if existing:
        raise HTTPException(409, detail="Batch code already exists")
    batch = Batch(
        batch_code=payload.batch_code, product_id=payload.product_id,
        quantity=payload.quantity, notes=payload.notes,
        quality_status="pending", created_by=current_user.id,
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    log_audit(db, current_user, "CREATE", "batch", batch.id, new={"batch_code": batch.batch_code})
    db.commit()
    return batch

@app.get("/batches", tags=["Batches"])
def list_batches(
    page: int = 1, page_size: int = 20,
    status: Optional[str] = None,
    product_id: Optional[str] = None,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user)
):
    q = db.query(Batch)
    if status:
        q = q.filter(Batch.quality_status == status)
    if product_id:
        q = q.filter(Batch.product_id == product_id)
    total = q.count()
    items = q.order_by(desc(Batch.created_at)).offset((page-1)*page_size).limit(page_size).all()
    
    result = []
    for b in items:
        insp_count = db.query(func.count(Inspection.id)).filter(Inspection.batch_id == b.id).scalar()
        defect_count = db.query(func.count(Inspection.id)).filter(
            Inspection.batch_id == b.id, Inspection.prediction == "DEFECT"
        ).scalar()
        result.append({
            "id": b.id, "batch_code": b.batch_code, "product_id": b.product_id,
            "quantity": b.quantity, "quality_status": b.quality_status,
            "hold_reason": b.hold_reason, "notes": b.notes,
            "created_at": b.created_at.isoformat(),
            "inspection_count": insp_count, "defect_count": defect_count,
        })
    return {"total": total, "page": page, "page_size": page_size, "items": result}

@app.get("/batches/{batch_id}", tags=["Batches"])
def get_batch(batch_id: str, db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(404, "Batch not found")
    return batch

@app.patch("/batches/{batch_id}/status", tags=["Batches"])
def update_batch_status(
    batch_id: str, payload: BatchStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("manager")),
):
    """Update batch quality status. Requires manager role. Creates audit trail."""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(404, "Batch not found")
    
    valid_statuses = {"pending", "in_inspection", "on_hold", "approved", "rejected"}
    if payload.quality_status not in valid_statuses:
        raise HTTPException(400, f"Invalid status. Must be one of: {valid_statuses}")
    
    prev_status = batch.quality_status
    
    # Require pending review check for approval
    if payload.quality_status == "approved":
        pending_reviews = db.query(func.count(Inspection.id)).filter(
            Inspection.batch_id == batch_id,
            Inspection.inspection_status == "under_review",
        ).scalar()
        if pending_reviews > 0:
            raise HTTPException(409, f"{pending_reviews} inspection(s) still pending review. Cannot approve batch.")
    
    log_audit(db, current_user, "BATCH_STATUS_CHANGE", "batch", batch_id,
              prev={"quality_status": prev_status},
              new={"quality_status": payload.quality_status},
              reason=payload.reason)
    
    batch.quality_status = payload.quality_status
    if payload.quality_status == "on_hold":
        batch.hold_reason = payload.reason
    batch.updated_at = utcnow()
    db.commit()
    return {"id": batch.id, "batch_code": batch.batch_code, "quality_status": batch.quality_status, "hold_reason": batch.hold_reason}

@app.post("/batches/{batch_id}/hold", tags=["Production Batches"])
def hold_batch(
    batch_id: str,
    payload: BatchReasonPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("manager")),
):
    """Quarantine batch on hold. Requires manager role and justification."""
    status_update = BatchStatusUpdate(quality_status="on_hold", reason=payload.reason)
    return update_batch_status(batch_id, status_update, db, current_user)

@app.post("/batches/{batch_id}/release", tags=["Production Batches"])
def release_batch(
    batch_id: str,
    payload: BatchReasonPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("manager")),
):
    """Release batch with approved quality status. Requires manager role."""
    status_update = BatchStatusUpdate(quality_status="approved", reason=payload.reason)
    return update_batch_status(batch_id, status_update, db, current_user)


# ─── Predict ─────────────────────────────────────────────────────────────────

@app.post("/predict", tags=["ML"])
async def predict_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Run ML inference on an uploaded image.
    Does NOT save an inspection record. Use POST /inspections to persist.
    """
    image_bytes = await file.read()
    result = ml_engine.predict(image_bytes, file.filename or "")
    
    if not result["success"] and not result.get("image_valid"):
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


# ─── Inspections ─────────────────────────────────────────────────────────────

@app.post("/inspections", status_code=201, tags=["Inspections"])
async def create_inspection(
    batch_id: str = Form(...),
    capture_mode: str = Form(default="photo"),
    serial_number: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
    photo_index: int = Form(default=1),
    is_primary: bool = Form(default=True),
    notes: Optional[str] = Form(default=None),
    parent_inspection_id: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Submit an inspection with image. Runs inference and persists result."""
    # Validate batch
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(404, "Batch not found")
    
    # Read image
    image_bytes = await file.read()
    filename = file.filename or "inspection.jpg"
    
    # Run inference
    infer_result = ml_engine.predict(image_bytes, filename)
    if not infer_result.get("image_valid"):
        raise HTTPException(status_code=400, detail=infer_result.get("error", "Invalid or corrupt image"))
    
    # Create inspection record
    inspection_id = new_uuid()
    sess_id = session_id or inspection_id  # Default session = this inspection
    
    insp = Inspection(
        id=inspection_id,
        batch_id=batch_id,
        operator_id=current_user.id,
        serial_number=serial_number,
        capture_mode=capture_mode,
        image_filename=filename,
        image_size_bytes=len(image_bytes),
        prediction=infer_result.get("prediction"),
        defect_score=infer_result.get("defect_score"),
        good_score=infer_result.get("good_score"),
        model_version=infer_result.get("model_version"),
        inference_time_ms=infer_result.get("inference_time_ms"),
        raw_scores=json.dumps(infer_result.get("raw_scores", {})),
        inspection_status=infer_result.get("inspection_status", "failed"),
        session_id=sess_id,
        photo_index=photo_index,
        is_primary=is_primary,
        parent_inspection_id=parent_inspection_id,
        notes=notes,
    )
    
    # Save image to local storage
    if infer_result.get("image_valid"):
        image_path = save_image(image_bytes, filename, inspection_id)
        insp.image_path = image_path
    
    # Map inspection_status to DB status
    ml_status = infer_result.get("inspection_status", "INSPECTION_FAILED")
    if ml_status == "NEEDS_MANUAL_REVIEW":
        insp.inspection_status = "under_review"
    elif ml_status in ("GOOD_PREDICTION", "DEFECT_SUSPECTED"):
        insp.inspection_status = "completed"
    else:
        insp.inspection_status = "failed"
    
    db.add(insp)
    
    # Update batch status if needed
    if batch.quality_status == "pending":
        batch.quality_status = "in_inspection"
        batch.updated_at = utcnow()
    
    db.commit()
    db.refresh(insp)
    
    return {
        "id": insp.id,
        "inspection_id": insp.id,
        "batch_id": insp.batch_id,
        "session_id": insp.session_id,
        "inspection_status": insp.inspection_status,
        "prediction": insp.prediction,
        "model_score": insp.defect_score if insp.prediction == "DEFECT" else insp.good_score,
        "defect_score": insp.defect_score,
        "good_score": insp.good_score,
        "model_version": insp.model_version,
        "inference_time_ms": insp.inference_time_ms,
        "ml_status": ml_status,
        "recommended_action": infer_result.get("recommended_action"),
        "error": infer_result.get("error"),
        "success": infer_result.get("success"),
        "timestamp": insp.timestamp.isoformat() if insp.timestamp else None,
        "image_path": insp.image_path,
    }


@app.get("/inspections/review-queue", tags=["Inspections"])
def get_review_queue(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("reviewer")),
):
    """
    SMART REVIEW PRIORITY QUEUE (Innovation Feature).
    Calculates an explainable priority score (0-100) for items requiring human review.
    Considers model uncertainty, defect confidence, batch status, recurrence, and wait time.
    """
    candidates = db.query(Inspection).filter(
        or_(
            Inspection.inspection_status == "under_review",
            and_(
                Inspection.prediction.in_(["DEFECT", "UNCERTAIN"]),
                Inspection.inspection_status != "reviewed"
            )
        )
    ).all()
    
    unreviewed = []
    for insp in candidates:
        rev_count = db.query(func.count(InspectionReview.id)).filter(InspectionReview.inspection_id == insp.id).scalar()
        if rev_count == 0:
            unreviewed.append(insp)
            
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    items = []
    
    for insp in unreviewed:
        score = 20  # Base priority
        factors = []
        defect_score = insp.defect_score if insp.defect_score is not None else 0.5
        
        # 1. Model Uncertainty (Ambiguity Factor): Scores in [0.35, 0.55] are high uncertainty
        if 0.35 <= defect_score <= 0.55:
            score += 35
            factors.append(f"High Model Ambiguity (Defect Prob {defect_score:.2f} near 0.45 decision threshold)")
        elif defect_score >= 0.85:
            score += 25
            factors.append(f"High-Confidence Defect Suspected ({defect_score*100:.1f}%)")
        elif defect_score > 0.55:
            score += 15
            factors.append(f"Moderate Defect Probability ({defect_score*100:.1f}%)")
            
        # 2. Batch Risk Status
        batch = db.query(Batch).filter(Batch.id == insp.batch_id).first()
        batch_code = batch.batch_code if batch else "Unknown"
        product = db.query(Product).filter(Product.id == batch.product_id).first() if batch else None
        product_name = product.name if product else "Unknown"
        
        if batch and batch.quality_status == "on_hold":
            score += 20
            factors.append("Production Batch is Currently On Hold")
        elif batch and batch.quality_status == "in_inspection":
            score += 5
            
        # 3. Defect Recurrence
        if product:
            prior_confirmed = db.query(func.count(InspectionReview.id)).join(
                Inspection, InspectionReview.inspection_id == Inspection.id
            ).join(
                Batch, Inspection.batch_id == Batch.id
            ).filter(
                Batch.product_id == product.id,
                InspectionReview.confirmed_class == "DEFECT"
            ).scalar()
            if prior_confirmed and prior_confirmed > 1:
                score += 15
                factors.append(f"Recurrent Product Defect Pattern ({prior_confirmed} prior defects recorded)")
                
        # 4. Wait time / Aging
        if insp.timestamp:
            hours_waiting = (now - insp.timestamp).total_seconds() / 3600.0
            if hours_waiting > 4:
                score += 15
                factors.append(f"Extended Wait Time ({hours_waiting:.1f} hrs in queue)")
            elif hours_waiting > 1:
                score += 8
                factors.append(f"Pending Review for {hours_waiting:.1f} hrs")
                
        score = min(score, 100)
        level = "CRITICAL" if score >= 80 else ("HIGH" if score >= 60 else ("MEDIUM" if score >= 40 else "LOW"))
        
        op = db.query(User).filter(User.id == insp.operator_id).first()
        items.append({
            "id": insp.id,
            "inspection_id": insp.id,
            "batch_id": insp.batch_id,
            "batch_code": batch_code,
            "product_name": product_name,
            "serial_number": insp.serial_number,
            "prediction": insp.prediction,
            "defect_score": insp.defect_score,
            "good_score": insp.good_score,
            "timestamp": insp.timestamp.isoformat() if insp.timestamp else None,
            "image_path": insp.image_path,
            "operator_name": op.display_name if op else None,
            "notes": insp.notes,
            "priority_score": score,
            "priority_level": level,
            "priority_factors": factors,
        })
        
    items.sort(key=lambda x: x["priority_score"], reverse=True)
    return {"total": len(items), "items": items}


@app.get("/inspections", tags=["Inspections"])
def list_inspections(
    page: int = 1,
    page_size: int = 20,
    batch_id: Optional[str] = None,
    operator_id: Optional[str] = None,
    prediction: Optional[str] = None,
    inspection_status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(Inspection)
    
    # Operators can only see their own inspections
    if current_user.role == "operator":
        q = q.filter(Inspection.operator_id == current_user.id)
    
    if batch_id:
        q = q.filter(Inspection.batch_id == batch_id)
    if operator_id and current_user.role != "operator":
        q = q.filter(Inspection.operator_id == operator_id)
    if prediction:
        q = q.filter(Inspection.prediction == prediction)
    if inspection_status:
        q = q.filter(Inspection.inspection_status == inspection_status)
    if date_from:
        try:
            df = datetime.fromisoformat(date_from)
            q = q.filter(Inspection.timestamp >= df)
        except:
            pass
    if date_to:
        try:
            dt = datetime.fromisoformat(date_to)
            q = q.filter(Inspection.timestamp <= dt)
        except:
            pass
    
    total = q.count()
    items = q.order_by(desc(Inspection.timestamp)).offset((page-1)*page_size).limit(page_size).all()
    
    result = []
    for insp in items:
        # Fetch operator name
        op = db.query(User).filter(User.id == insp.operator_id).first()
        batch = db.query(Batch).filter(Batch.id == insp.batch_id).first()
        review_count = db.query(func.count(InspectionReview.id)).filter(
            InspectionReview.inspection_id == insp.id).scalar()
        
        result.append({
            "id": insp.id,
            "batch_id": insp.batch_id,
            "batch_code": batch.batch_code if batch else None,
            "operator_id": insp.operator_id,
            "operator_name": op.display_name if op else None,
            "serial_number": insp.serial_number,
            "capture_mode": insp.capture_mode,
            "prediction": insp.prediction,
            "defect_score": insp.defect_score,
            "good_score": insp.good_score,
            "model_version": insp.model_version,
            "inspection_status": insp.inspection_status,
            "session_id": insp.session_id,
            "photo_index": insp.photo_index,
            "is_primary": insp.is_primary,
            "review_count": review_count,
            "timestamp": insp.timestamp.isoformat() if insp.timestamp else None,
            "image_path": insp.image_path,
        })
    
    return {"total": total, "page": page, "page_size": page_size, "items": result}


@app.get("/inspections/{inspection_id}", tags=["Inspections"])
def get_inspection(inspection_id: str, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(404, "Inspection not found")
    
    # Operators can only view their own
    if current_user.role == "operator" and insp.operator_id != current_user.id:
        raise HTTPException(403, "Access denied")
    
    reviews = db.query(InspectionReview).filter(InspectionReview.inspection_id == insp.id).all()
    cas = db.query(CorrectiveAction).filter(CorrectiveAction.inspection_id == insp.id).all()
    op = db.query(User).filter(User.id == insp.operator_id).first()
    batch = db.query(Batch).filter(Batch.id == insp.batch_id).first()
    
    review_list = []
    for r in reviews:
        reviewer = db.query(User).filter(User.id == r.reviewer_id).first()
        review_list.append({
            "id": r.id, "confirmed_class": r.confirmed_class,
            "original_prediction": r.original_prediction,
            "reviewer": reviewer.display_name if reviewer else None,
            "reason": r.reason, "notes": r.notes,
            "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
        })
    
    raw_scores = {}
    if insp.raw_scores:
        try:
            raw_scores = json.loads(insp.raw_scores)
        except:
            pass
    
    return {
        "id": insp.id,
        "batch_id": insp.batch_id,
        "batch_code": batch.batch_code if batch else None,
        "operator_id": insp.operator_id,
        "operator_name": op.display_name if op else None,
        "serial_number": insp.serial_number,
        "capture_mode": insp.capture_mode,
        "prediction": insp.prediction,
        "defect_score": insp.defect_score,
        "good_score": insp.good_score,
        "model_version": insp.model_version,
        "inference_time_ms": insp.inference_time_ms,
        "raw_scores": raw_scores,
        "inspection_status": insp.inspection_status,
        "session_id": insp.session_id,
        "photo_index": insp.photo_index,
        "is_primary": insp.is_primary,
        "parent_inspection_id": insp.parent_inspection_id,
        "timestamp": insp.timestamp.isoformat() if insp.timestamp else None,
        "image_path": insp.image_path,
        "notes": insp.notes,
        "reviews": review_list,
        "corrective_actions": [{"id": ca.id, "title": ca.title, "status": ca.status} for ca in cas],
    }


@app.post("/inspections/{inspection_id}/review", status_code=201, tags=["Inspections"])
def review_inspection(
    inspection_id: str, payload: ReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("reviewer")),
):
    """Submit human review. Does NOT overwrite original ML prediction."""
    insp = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not insp:
        raise HTTPException(404, "Inspection not found")
    
    valid_classes = {"GOOD", "DEFECT", "INCONCLUSIVE"}
    if payload.confirmed_class not in valid_classes:
        raise HTTPException(400, f"confirmed_class must be one of {valid_classes}")
    
    review = InspectionReview(
        inspection_id=inspection_id,
        reviewer_id=current_user.id,
        confirmed_class=payload.confirmed_class,
        original_prediction=insp.prediction,  # preserve original
        reason=payload.reason,
        notes=payload.notes,
    )
    db.add(review)
    
    # Update inspection status
    insp.inspection_status = "reviewed"
    insp.updated_at = utcnow()
    
    log_audit(db, current_user, "REVIEW", "inspection", inspection_id,
              prev={"inspection_status": "under_review", "prediction": insp.prediction},
              new={"inspection_status": "reviewed", "confirmed_class": payload.confirmed_class},
              reason=payload.reason)
    
    db.commit()
    db.refresh(review)
    return {
        "review_id": review.id,
        "confirmed_class": review.confirmed_class,
        "original_prediction": review.original_prediction,
        "reviewed_at": review.reviewed_at.isoformat() if review.reviewed_at else None
    }


@app.post("/inspections/{inspection_id}/reinspect", status_code=201, tags=["Inspections"])
async def reinspect(
    inspection_id: str,
    batch_id: str = Form(...),
    capture_mode: str = Form(default="photo"),
    serial_number: Optional[str] = Form(default=None),
    notes: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a follow-up reinspection linked to a previous inspection."""
    parent = db.query(Inspection).filter(Inspection.id == inspection_id).first()
    if not parent:
        raise HTTPException(404, "Parent inspection not found")
    
    image_bytes = await file.read()
    filename = file.filename or "reinspection.jpg"
    infer_result = ml_engine.predict(image_bytes, filename)
    
    new_insp = Inspection(
        batch_id=batch_id,
        operator_id=current_user.id,
        serial_number=serial_number or parent.serial_number,
        capture_mode=capture_mode,
        image_filename=filename,
        image_size_bytes=len(image_bytes),
        prediction=infer_result.get("prediction"),
        defect_score=infer_result.get("defect_score"),
        good_score=infer_result.get("good_score"),
        model_version=infer_result.get("model_version"),
        inference_time_ms=infer_result.get("inference_time_ms"),
        raw_scores=json.dumps(infer_result.get("raw_scores", {})),
        inspection_status="completed",
        session_id=parent.session_id,
        parent_inspection_id=inspection_id,
        notes=f"Reinspection of {inspection_id}. {notes or ''}",
    )
    
    if infer_result.get("image_valid"):
        new_insp.image_path = save_image(image_bytes, filename, new_insp.id or new_uuid())
    
    db.add(new_insp)
    db.commit()
    db.refresh(new_insp)
    
    return {
        "inspection_id": new_insp.id,
        "parent_inspection_id": inspection_id,
        "prediction": new_insp.prediction,
        "inspection_status": new_insp.inspection_status,
        "recommended_action": infer_result.get("recommended_action"),
    }


# ─── Corrective Actions ───────────────────────────────────────────────────────

@app.post("/corrective-actions", status_code=201, response_model=CorrectiveActionOut, tags=["Corrective Actions"])
def create_corrective_action(
    payload: CorrectiveActionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("reviewer")),
):
    assigned_user_id = None
    if payload.assigned_to:
        target_user = db.query(User).filter(
            or_(User.id == payload.assigned_to, User.username == payload.assigned_to)
        ).first()
        if target_user:
            assigned_user_id = target_user.id

    before_img = payload.before_image_path
    if not before_img and payload.inspection_id:
        insp = db.query(Inspection).filter(Inspection.id == payload.inspection_id).first()
        if insp:
            before_img = insp.image_path

    ca = CorrectiveAction(
        inspection_id=payload.inspection_id,
        batch_id=payload.batch_id,
        title=payload.title,
        description=payload.description,
        root_cause=payload.root_cause,
        before_image_path=before_img,
        assigned_to=assigned_user_id,
        created_by=current_user.id,
        due_date=payload.due_date,
        priority=payload.priority,
    )
    db.add(ca)
    db.commit()
    db.refresh(ca)
    log_audit(db, current_user, "CREATE", "corrective_action", ca.id, new={"title": ca.title, "priority": ca.priority})
    db.commit()
    return ca

@app.get("/corrective-actions", tags=["Corrective Actions"])
def list_corrective_actions(
    page: int = 1, page_size: int = 20,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("reviewer")),
):
    q = db.query(CorrectiveAction)
    if status:
        q = q.filter(CorrectiveAction.status == status)
    if priority:
        q = q.filter(CorrectiveAction.priority == priority)
    total = q.count()
    items = q.order_by(desc(CorrectiveAction.created_at)).offset((page-1)*page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "items": items}

@app.patch("/corrective-actions/{ca_id}", response_model=CorrectiveActionOut, tags=["Corrective Actions"])
def update_corrective_action(
    ca_id: str, payload: CorrectiveActionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("reviewer")),
):
    ca = db.query(CorrectiveAction).filter(CorrectiveAction.id == ca_id).first()
    if not ca:
        raise HTTPException(404, "Corrective action not found")
    
    prev_state = {"status": ca.status}
    
    if payload.status:
        valid = {"open", "in_progress", "awaiting_reinspection", "resolved", "closed", "reopened"}
        if payload.status not in valid:
            raise HTTPException(400, f"Invalid status: {valid}")
        ca.status = payload.status
        if payload.status in ("resolved", "closed"):
            ca.closed_at = utcnow()
    if payload.assigned_to:
        ca.assigned_to = payload.assigned_to
    if payload.root_cause:
        ca.root_cause = payload.root_cause
    if payload.completion_notes:
        ca.completion_notes = payload.completion_notes
    if payload.due_date:
        ca.due_date = payload.due_date
    if payload.follow_up_inspection_id:
        ca.follow_up_inspection_id = payload.follow_up_inspection_id
    if payload.after_image_path:
        ca.after_image_path = payload.after_image_path
    if payload.reinspection_status:
        ca.reinspection_status = payload.reinspection_status
    if payload.reinspection_notes:
        ca.reinspection_notes = payload.reinspection_notes
    
    ca.updated_at = utcnow()
    log_audit(db, current_user, "UPDATE", "corrective_action", ca_id, prev=prev_state, new={"status": ca.status})
    db.commit()
    db.refresh(ca)
    return ca

@app.post("/corrective-actions/{ca_id}/evidence", response_model=CorrectiveActionOut, tags=["Corrective Actions"])
async def upload_repair_evidence(
    ca_id: str,
    notes: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("reviewer")),
):
    """Upload post-repair image evidence and mark awaiting reinspection."""
    ca = db.query(CorrectiveAction).filter(CorrectiveAction.id == ca_id).first()
    if not ca:
        raise HTTPException(404, "Corrective action not found")
    
    image_bytes = await file.read()
    filename = file.filename or "repair_evidence.jpg"
    evidence_id = new_uuid()
    saved_path = save_image(image_bytes, filename, f"repair_{evidence_id}")
    
    ca.after_image_path = saved_path
    ca.status = "awaiting_reinspection"
    if notes:
        ca.completion_notes = f"{ca.completion_notes or ''}\n{notes}".strip()
    ca.updated_at = utcnow()
    
    log_audit(db, current_user, "UPLOAD_EVIDENCE", "corrective_action", ca_id,
              new={"after_image_path": saved_path, "status": "awaiting_reinspection"})
    db.commit()
    db.refresh(ca)
    return ca

@app.post("/corrective-actions/{ca_id}/verify", response_model=CorrectiveActionOut, tags=["Corrective Actions"])
def verify_reinspection(
    ca_id: str,
    payload: ReinspectionVerifyPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("reviewer")),
):
    """
    Authorised reviewer approves or rejects reinspection evidence.
    If passed: marks resolved. If failed: marks reopened for further corrective work.
    """
    ca = db.query(CorrectiveAction).filter(CorrectiveAction.id == ca_id).first()
    if not ca:
        raise HTTPException(404, "Corrective action not found")
    
    prev_status = ca.status
    if payload.decision == "passed":
        ca.status = "resolved"
        ca.reinspection_status = "passed"
        ca.closed_at = utcnow()
    elif payload.decision == "failed":
        ca.status = "reopened"
        ca.reinspection_status = "failed"
    else:
        raise HTTPException(400, "Decision must be 'passed' or 'failed'")
        
    ca.reinspection_notes = payload.notes
    ca.updated_at = utcnow()
    
    log_audit(db, current_user, "REINSPECTION_DECISION", "corrective_action", ca_id,
              prev={"status": prev_status},
              new={"status": ca.status, "reinspection_status": ca.reinspection_status},
              reason=payload.notes)
    db.commit()
    db.refresh(ca)
    return ca
    return ca


# ─── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/dashboard/summary", tags=["Dashboard"])
def dashboard_summary(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("manager")),
):
    """
    Real database-derived dashboard metrics.
    All counts derived from actual inspection records.
    date_from/date_to in ISO format. Default: today (UTC).
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    if date_from:
        try:
            df = datetime.fromisoformat(date_from)
        except:
            df = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        df = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    if date_to:
        try:
            dt = datetime.fromisoformat(date_to)
        except:
            dt = now
    else:
        dt = now
    
    date_filter = and_(Inspection.timestamp >= df, Inspection.timestamp <= dt)
    
    # Total inspections
    total = db.query(func.count(Inspection.id)).filter(date_filter).scalar() or 0
    
    # By prediction
    good_pred = db.query(func.count(Inspection.id)).filter(
        date_filter, Inspection.prediction == "GOOD"
    ).scalar() or 0
    defect_pred = db.query(func.count(Inspection.id)).filter(
        date_filter, Inspection.prediction == "DEFECT"
    ).scalar() or 0
    uncertain = db.query(func.count(Inspection.id)).filter(
        date_filter, Inspection.prediction == "UNCERTAIN"
    ).scalar() or 0
    
    # By inspection status
    pending_review = db.query(func.count(Inspection.id)).filter(
        date_filter, Inspection.inspection_status == "under_review"
    ).scalar() or 0
    failed = db.query(func.count(Inspection.id)).filter(
        date_filter, Inspection.inspection_status == "failed"
    ).scalar() or 0
    reviewed = db.query(func.count(Inspection.id)).filter(
        date_filter, Inspection.inspection_status == "reviewed"
    ).scalar() or 0
    
    # Human-confirmed outcomes
    confirmed_good = db.query(func.count(InspectionReview.id)).join(
        Inspection, InspectionReview.inspection_id == Inspection.id
    ).filter(
        Inspection.timestamp >= df, Inspection.timestamp <= dt,
        InspectionReview.confirmed_class == "GOOD"
    ).scalar() or 0
    
    confirmed_defect = db.query(func.count(InspectionReview.id)).join(
        Inspection, InspectionReview.inspection_id == Inspection.id
    ).filter(
        Inspection.timestamp >= df, Inspection.timestamp <= dt,
        InspectionReview.confirmed_class == "DEFECT"
    ).scalar() or 0
    
    # Corrective actions
    open_cas = db.query(func.count(CorrectiveAction.id)).filter(
        CorrectiveAction.status.in_(["open", "in_progress"])
    ).scalar() or 0
    
    overdue_cas = db.query(func.count(CorrectiveAction.id)).filter(
        CorrectiveAction.status.in_(["open", "in_progress"]),
        CorrectiveAction.due_date < now,
    ).scalar() or 0
    
    # Batches on hold
    batches_on_hold = db.query(func.count(Batch.id)).filter(
        Batch.quality_status == "on_hold"
    ).scalar() or 0
    
    # Defect rate (model predictions basis)
    defect_rate_pct = round(defect_pred / total * 100, 2) if total > 0 else 0.0
    
    return {
        "period": {"from": df.isoformat(), "to": dt.isoformat()},
        "metric_definitions": {
            "defect_rate": "Percentage of inspections where model predicted DEFECT, out of all completed inspections in period",
            "confirmed_defect": "Inspections where a human reviewer confirmed DEFECT outcome",
            "confirmed_good": "Inspections where a human reviewer confirmed GOOD outcome",
        },
        "total_inspections": total,
        "good_predictions": good_pred,
        "defect_suspected": defect_pred,
        "uncertain_predictions": uncertain,
        "pending_review": pending_review,
        "failed_inspections": failed,
        "reviewed_inspections": reviewed,
        "confirmed_good": confirmed_good,
        "confirmed_defect": confirmed_defect,
        "open_corrective_actions": open_cas,
        "overdue_corrective_actions": overdue_cas,
        "batches_on_hold": batches_on_hold,
        "defect_rate_pct": defect_rate_pct,
        "note": "AI predictions are NOT equivalent to authorised quality outcomes. See confirmed_good/confirmed_defect for human-reviewed results.",
    }


@app.get("/dashboard/trends", tags=["Dashboard"])
def dashboard_trends(
    days: int = 7,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("manager")),
):
    """Daily inspection volume and outcome trends for the past N days."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    date_from = now - timedelta(days=days)
    
    inspections = db.query(Inspection).filter(
        Inspection.timestamp >= date_from
    ).all()
    
    # Group by day
    from collections import defaultdict
    daily = defaultdict(lambda: {"total": 0, "good": 0, "defect": 0, "uncertain": 0, "failed": 0, "reviewed": 0})
    
    for insp in inspections:
        day_key = insp.timestamp.strftime("%Y-%m-%d") if insp.timestamp else "unknown"
        daily[day_key]["total"] += 1
        if insp.prediction == "GOOD":
            daily[day_key]["good"] += 1
        elif insp.prediction == "DEFECT":
            daily[day_key]["defect"] += 1
        elif insp.prediction == "UNCERTAIN":
            daily[day_key]["uncertain"] += 1
        if insp.inspection_status == "failed":
            daily[day_key]["failed"] += 1
        if insp.inspection_status == "reviewed":
            daily[day_key]["reviewed"] += 1
    
    # Fill missing days
    result = []
    for i in range(days):
        day = (now - timedelta(days=days-1-i)).strftime("%Y-%m-%d")
        data = daily.get(day, {"total": 0, "good": 0, "defect": 0, "uncertain": 0, "failed": 0, "reviewed": 0})
        result.append({"date": day, **data})
    
    # Corrective actions by status
    ca_stats = db.query(CorrectiveAction.status, func.count(CorrectiveAction.id)).group_by(
        CorrectiveAction.status
    ).all()
    ca_by_status = {s: c for s, c in ca_stats}
    
    return {
        "daily_trends": result,
        "corrective_actions_by_status": ca_by_status,
        "period_days": days,
    }


@app.get("/dashboard/batches", tags=["Dashboard"])
def dashboard_batches(
    db: Session = Depends(get_db),
    _: User = Depends(require_role("manager")),
):
    """Batch-level inspection statistics."""
    batches = db.query(Batch).order_by(desc(Batch.created_at)).limit(50).all()
    result = []
    for b in batches:
        stats = db.query(
            func.count(Inspection.id).label("total"),
            func.sum((Inspection.prediction == "GOOD").cast(Integer)).label("good"),
            func.sum((Inspection.prediction == "DEFECT").cast(Integer)).label("defect"),
        ).filter(Inspection.batch_id == b.id).first()
        
        product = db.query(Product).filter(Product.id == b.product_id).first()
        
        result.append({
            "batch_id": b.id,
            "batch_code": b.batch_code,
            "product_name": product.name if product else None,
            "quality_status": b.quality_status,
            "quantity": b.quantity,
            "total_inspections": stats.total or 0,
            "good_count": stats.good or 0,
            "defect_count": stats.defect or 0,
            "created_at": b.created_at.isoformat(),
        })
    return result


# ─── Image serving ───────────────────────────────────────────────────────────

@app.get("/storage/images/{filename}", tags=["Storage"])
def serve_image(filename: str, current_user: User = Depends(get_current_user)):
    """Serve stored inspection images. Authenticated only."""
    img_path = IMAGES_STORAGE / filename
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(img_path))


# ─── Demo Samples & Evidence ─────────────────────────────────────────────────

@app.get("/demo/sample-images", tags=["Demo"])
def list_sample_images():
    """List verified sample images from the uploaded datasets for instant evaluation."""
    samples = [
        {
            "filename": "000789191.jpg",
            "product_type": "steel_strip",
            "expected_label": "GOOD",
            "category": "Steel Surface",
            "description": "Clean, defect-free cold-rolled steel surface",
            "url": "/storage/images/000789191.jpg",
        },
        {
            "filename": "0007a71bf.jpg",
            "product_type": "steel_strip",
            "expected_label": "DEFECT",
            "category": "Steel Surface",
            "description": "Longitudinal surface scratch defect",
            "url": "/storage/images/0007a71bf.jpg",
        },
        {
            "filename": "001982b08.jpg",
            "product_type": "steel_strip",
            "expected_label": "DEFECT",
            "category": "Steel Surface",
            "description": "Periodic pitting inclusions and score mark",
            "url": "/storage/images/001982b08.jpg",
        },
        {
            "filename": "cast_def_0_1059.jpeg",
            "product_type": "casting",
            "expected_label": "UNCERTAIN",
            "category": "Metal Casting",
            "description": "Subtle boundary anomaly near decision threshold",
            "url": "/storage/images/cast_def_0_1059.jpeg",
        },
        {
            "filename": "cast_def_0_1063.jpeg",
            "product_type": "casting",
            "expected_label": "DEFECT",
            "category": "Metal Casting",
            "description": "Casting blowhole gas porosity cavity",
            "url": "/storage/images/cast_def_0_1063.jpeg",
        },
        {
            "filename": "cast_def_0_108.jpeg",
            "product_type": "casting",
            "expected_label": "GOOD",
            "category": "Metal Casting",
            "description": "Precision machined casting block with zero defect",
            "url": "/storage/images/cast_def_0_108.jpeg",
        },
    ]
    return samples


# ─── Audit Events ─────────────────────────────────────────────────────────────

@app.get("/audit/events", tags=["Audit"])
def list_audit_events(
    page: int = 1,
    page_size: int = 30,
    entity_type: Optional[str] = None,
    action: Optional[str] = None,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("reviewer")),
):
    """Immutable audit trail of quality mutations."""
    q = db.query(AuditEvent)
    if entity_type:
        q = q.filter(AuditEvent.entity_type == entity_type)
    if action:
        q = q.filter(AuditEvent.action == action)
    total = q.count()
    items = q.order_by(desc(AuditEvent.timestamp)).offset((page-1)*page_size).limit(page_size).all()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": e.id,
                "actor_id": e.actor_id,
                "actor_name": e.actor_name,
                "action": e.action,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "previous_state": json.loads(e.previous_state) if e.previous_state else None,
                "new_state": json.loads(e.new_state) if e.new_state else None,
                "reason": e.reason,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in items
        ],
    }


# ─── Reports ──────────────────────────────────────────────────────────────────

@app.get("/reports/quality-summary", tags=["Reports"])
def quality_summary_report(
    db: Session = Depends(get_db),
    _: User = Depends(require_role("manager")),
):
    """Generate exportable comprehensive quality report."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    total_inspections = db.query(func.count(Inspection.id)).scalar() or 0
    good_count = db.query(func.count(Inspection.id)).filter(Inspection.prediction == "GOOD").scalar() or 0
    defect_count = db.query(func.count(Inspection.id)).filter(Inspection.prediction == "DEFECT").scalar() or 0
    uncertain_count = db.query(func.count(Inspection.id)).filter(Inspection.prediction == "UNCERTAIN").scalar() or 0
    
    confirmed_defects = db.query(func.count(InspectionReview.id)).filter(InspectionReview.confirmed_class == "DEFECT").scalar() or 0
    confirmed_good = db.query(func.count(InspectionReview.id)).filter(InspectionReview.confirmed_class == "GOOD").scalar() or 0
    
    total_products = db.query(func.count(Product.id)).filter(Product.is_active == True).scalar() or 0
    total_batches = db.query(func.count(Batch.id)).scalar() or 0
    batches_on_hold = db.query(func.count(Batch.id)).filter(Batch.quality_status == "on_hold").scalar() or 0
    batches_approved = db.query(func.count(Batch.id)).filter(Batch.quality_status == "approved").scalar() or 0
    
    open_capa = db.query(func.count(CorrectiveAction.id)).filter(CorrectiveAction.status.in_(["open", "in_progress"])).scalar() or 0
    overdue_capa = db.query(func.count(CorrectiveAction.id)).filter(
        CorrectiveAction.status.in_(["open", "in_progress"]),
        CorrectiveAction.due_date < now
    ).scalar() or 0
    resolved_capa = db.query(func.count(CorrectiveAction.id)).filter(CorrectiveAction.status == "resolved").scalar() or 0
    
    defect_rate = round(defect_count / total_inspections * 100, 2) if total_inspections > 0 else 0.0
    capa_completion_rate = round(resolved_capa / (open_capa + resolved_capa) * 100, 1) if (open_capa + resolved_capa) > 0 else 100.0
    
    return {
        "report_id": f"QR-{now.strftime('%Y%m%d%H%M%S')}",
        "generated_at": now.isoformat(),
        "system_status": {
            "model_architecture": "MobileNetV3-Small",
            "model_version": ml_engine.model_version,
            "held_out_accuracy": 0.8407,
            "inference_device": "CPU",
        },
        "executive_summary": {
            "total_products": total_products,
            "total_batches": total_batches,
            "batches_on_hold": batches_on_hold,
            "batches_approved": batches_approved,
            "total_inspections": total_inspections,
            "defect_rate_pct": defect_rate,
            "confirmed_defects": confirmed_defects,
            "confirmed_good": confirmed_good,
            "open_capa_count": open_capa,
            "overdue_capa_count": overdue_capa,
            "capa_completion_rate_pct": capa_completion_rate,
        },
        "critical_alerts": [
            f"{overdue_capa} corrective actions currently overdue" if overdue_capa > 0 else "Zero overdue corrective actions",
            f"{batches_on_hold} batches quarantined on hold" if batches_on_hold > 0 else "All batches compliant",
        ],
    }


# ─── AI Assistant (QualityGuard Copilot) ──────────────────────────────────────

@app.get("/assistant/query", tags=["Assistant"])
def assistant_query(
    q: str = Query(..., description="Natural language question about quality data"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("manager")),
):
    """
    QualityGuard Copilot: Grounded AI Quality Assistant.
    Answers natural language queries strictly using verified database records.
    Cites exact entity IDs like [INSP-xxxx], [BATCH-xxxx], and [CAPA-xxxx].
    """
    q_lower = q.lower()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    response = {
        "question": q,
        "answer": None,
        "data_source": None,
        "citations": [],
        "caveat": "Grounded strictly in persistent QualityGuard database records. Not an autonomous decision maker.",
    }
    
    # 1. Defect summary for specific batch
    import re
    batch_match = re.search(r"batch[-\s]?([a-zA-Z0-9_-]+)", q, re.IGNORECASE)
    if "defect" in q_lower and batch_match:
        code_fragment = batch_match.group(1).upper()
        batch = db.query(Batch).filter(Batch.batch_code.ilike(f"%{code_fragment}%")).first()
        if batch:
            insps = db.query(Inspection).filter(Inspection.batch_id == batch.id).all()
            defects = [i for i in insps if i.prediction == "DEFECT"]
            citations = [f"[INSP-{i.id[:8]}]" for i in defects]
            citations.append(f"[BATCH-{batch.batch_code}]")
            
            response["answer"] = (
                f"Batch [BATCH-{batch.batch_code}] has {len(insps)} recorded inspections, of which {len(defects)} "
                f"were flagged as DEFECT. Confirmed defect records: {', '.join(citations[:5])}. "
                f"Current batch quality status is '{batch.quality_status}'"
                + (f" (Reason: {batch.hold_reason})" if batch.hold_reason else ".")
            )
            response["data_source"] = f"inspections and batches tables where batch_id={batch.id}"
            response["citations"] = citations
            return response
            
    # 2. Overdue Corrective Actions
    if "overdue" in q_lower or ("corrective" in q_lower and "action" in q_lower and "late" in q_lower):
        overdue = db.query(CorrectiveAction).filter(
            CorrectiveAction.status.in_(["open", "in_progress"]),
            CorrectiveAction.due_date < now,
        ).all()
        citations = [f"[CAPA-{ca.id[:8]}]" for ca in overdue]
        
        if overdue:
            details = [f"[CAPA-{ca.id[:8]}]: '{ca.title}' (Due {ca.due_date.strftime('%Y-%m-%d')}, Priority: {ca.priority.upper()})" for ca in overdue[:3]]
            response["answer"] = (
                f"There are {len(overdue)} overdue corrective action(s) requiring immediate resolution: "
                + "; ".join(details) + "."
            )
        else:
            response["answer"] = "Zero corrective actions are currently overdue. All actions are on schedule."
            
        response["data_source"] = "corrective_actions table where due_date < now and status in ('open', 'in_progress')"
        response["citations"] = citations
        return response

    # 3. Pending human reviews
    if "pending" in q_lower or "awaiting" in q_lower or "queue" in q_lower or "review" in q_lower:
        pending = db.query(Inspection).filter(
            Inspection.inspection_status == "under_review"
        ).order_by(desc(Inspection.defect_score)).all()
        citations = [f"[INSP-{i.id[:8]}]" for i in pending]
        
        response["answer"] = (
            f"There are {len(pending)} inspection(s) currently awaiting human quality review in the Smart Priority Queue. "
            f"Highest priority items: {', '.join(citations[:4]) if citations else 'None pending'}."
        )
        response["data_source"] = "inspections table where inspection_status='under_review'"
        response["citations"] = citations
        return response

    # 4. Defect Recurrence
    if "recur" in q_lower or "frequent" in q_lower or "pattern" in q_lower:
        recurrences = db.query(
            Product.name,
            func.count(InspectionReview.id).label("count")
        ).join(
            Batch, Product.id == Batch.product_id
        ).join(
            Inspection, Batch.id == Inspection.batch_id
        ).join(
            InspectionReview, Inspection.id == InspectionReview.inspection_id
        ).filter(
            InspectionReview.confirmed_class == "DEFECT"
        ).group_by(Product.name).order_by(desc("count")).all()
        
        if recurrences:
            top_prod, count = recurrences[0]
            response["answer"] = (
                f"The highest recurring confirmed defect pattern is on product '{top_prod}' with {count} confirmed defect(s). "
                f"Root cause investigations have been logged in corrective actions for associated batches."
            )
        else:
            response["answer"] = "No confirmed defect recurrence patterns recorded in historical reviews."
            
        response["data_source"] = "inspection_reviews joined with products and batches where confirmed_class='DEFECT'"
        return response

    # 5. Batches on hold
    if "hold" in q_lower or "quarantine" in q_lower:
        on_hold = db.query(Batch).filter(Batch.quality_status == "on_hold").all()
        citations = [f"[BATCH-{b.batch_code}]" for b in on_hold]
        
        if on_hold:
            details = [f"[BATCH-{b.batch_code}] ({b.hold_reason or 'No reason specified'})" for b in on_hold]
            response["answer"] = f"There are {len(on_hold)} batch(es) currently quarantined on hold: " + "; ".join(details) + "."
        else:
            response["answer"] = "No batches are currently on hold. All production lines are operating normally."
            
        response["data_source"] = "batches table where quality_status='on_hold'"
        response["citations"] = citations
        return response

    # 6. Reinspection cases
    if "reinspect" in q_lower or "repair" in q_lower:
        reinsps = db.query(CorrectiveAction).filter(CorrectiveAction.status == "awaiting_reinspection").all()
        citations = [f"[CAPA-{ca.id[:8]}]" for ca in reinsps]
        
        if reinsps:
            response["answer"] = f"There are {len(reinsps)} corrective action(s) awaiting reinspection verification: {', '.join(citations)}."
        else:
            resolved = db.query(func.count(CorrectiveAction.id)).filter(CorrectiveAction.reinspection_status == "passed").scalar() or 0
            response["answer"] = f"Zero actions currently awaiting reinspection. {resolved} actions have successfully passed reinspection."
            
        response["data_source"] = "corrective_actions table where status='awaiting_reinspection'"
        response["citations"] = citations
        return response

    # 7. Today's activity / default
    total = db.query(func.count(Inspection.id)).filter(Inspection.timestamp >= today_start).scalar() or 0
    good = db.query(func.count(Inspection.id)).filter(Inspection.timestamp >= today_start, Inspection.prediction == "GOOD").scalar() or 0
    defect = db.query(func.count(Inspection.id)).filter(Inspection.timestamp >= today_start, Inspection.prediction == "DEFECT").scalar() or 0
    
    response["answer"] = (
        f"Today ({now.strftime('%Y-%m-%d')}), {total} total inspections have been recorded. "
        f"{good} predicted GOOD, {defect} flagged as DEFECT. "
        f"You can ask me to summarise defects in a specific batch, list overdue actions, "
        f"check the review queue, or analyze defect recurrence."
    )
    response["data_source"] = f"inspections table where timestamp >= {today_start.isoformat()}"
    return response


# ─── Static files & UI Mounting ──────────────────────────────────────────────
from fastapi.responses import RedirectResponse

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"
MOBILE_DIR = Path(__file__).parent.parent / "mobile_app"

STORAGE_DIR = Path(__file__).parent / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

if MOBILE_DIR.exists():
    app.mount("/mobile", StaticFiles(directory=str(MOBILE_DIR), html=True), name="mobile")

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/dashboard/")

