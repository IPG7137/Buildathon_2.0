"""
Database layer for PS01 - SQLite with SQLAlchemy
=================================================
Uses SQLite for development (zero-config). 
Schema supports products, batches, inspections, reviews, corrective actions, audit log.
"""

import os
import json
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, Text,
    DateTime, ForeignKey, Index, event, text
)
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker
from sqlalchemy.pool import StaticPool

# ─── Database path ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.parent
DB_DIR = ROOT_DIR / "backend" / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "ps01.db"
IMAGES_STORAGE = ROOT_DIR / "backend" / "storage" / "images"
IMAGES_STORAGE.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ─── Engine & session ────────────────────────────────────────────────────────
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

# Enable WAL mode and foreign keys for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── Models ──────────────────────────────────────────────────────────────────

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

def new_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=new_uuid)
    username = Column(String(64), unique=True, nullable=False, index=True)
    display_name = Column(String(128), nullable=False)
    email = Column(String(256), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(32), nullable=False, default="operator")  # operator|reviewer|manager|admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    last_login = Column(DateTime, nullable=True)
    
    inspections = relationship("Inspection", back_populates="operator", foreign_keys="Inspection.operator_id")
    reviews = relationship("InspectionReview", back_populates="reviewer")
    corrective_actions_created = relationship("CorrectiveAction", back_populates="created_by_user", foreign_keys="CorrectiveAction.created_by")
    corrective_actions_assigned = relationship("CorrectiveAction", back_populates="assigned_user", foreign_keys="CorrectiveAction.assigned_to")


class Product(Base):
    __tablename__ = "products"
    
    id = Column(String(36), primary_key=True, default=new_uuid)
    name = Column(String(128), nullable=False)
    product_type = Column(String(64), nullable=False, default="steel_strip")  # steel_strip|casting|etc
    sku = Column(String(64), nullable=True)
    description = Column(Text, nullable=True)
    inspection_category = Column(String(64), default="surface_defect")
    inspection_criteria = Column(Text, nullable=True)
    threshold_defect = Column(Float, default=0.45)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    
    batches = relationship("Batch", back_populates="product")


class Batch(Base):
    __tablename__ = "batches"
    
    id = Column(String(36), primary_key=True, default=new_uuid)
    batch_code = Column(String(64), unique=True, nullable=False, index=True)
    product_id = Column(String(36), ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=True)
    production_line = Column(String(64), nullable=True, default="Line 1")
    production_date = Column(DateTime, nullable=True, default=utcnow)
    # Status: pending|in_inspection|on_hold|approved|rejected
    quality_status = Column(String(32), default="pending")
    hold_reason = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    product = relationship("Product", back_populates="batches")
    inspections = relationship("Inspection", back_populates="batch")


class Inspection(Base):
    __tablename__ = "inspections"
    
    id = Column(String(36), primary_key=True, default=new_uuid)
    batch_id = Column(String(36), ForeignKey("batches.id"), nullable=False)
    operator_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    serial_number = Column(String(64), nullable=True)       # optional item serial
    capture_mode = Column(String(32), default="photo")     # camera|photo|upload
    
    # Image reference
    image_path = Column(String(512), nullable=True)        # relative path under storage/images
    image_filename = Column(String(256), nullable=True)
    image_size_bytes = Column(Integer, nullable=True)
    
    # ML results
    prediction = Column(String(32), nullable=True)          # GOOD|DEFECT|UNCERTAIN
    defect_score = Column(Float, nullable=True)             # prob(DEFECT) [0,1]
    good_score = Column(Float, nullable=True)
    model_version = Column(String(32), nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    raw_scores = Column(Text, nullable=True)                # JSON
    
    # Status: pending|completed|failed|under_review|reviewed
    inspection_status = Column(String(32), default="pending")
    
    # Session info (for multi-photo evidence)
    session_id = Column(String(36), nullable=True, index=True)
    photo_index = Column(Integer, default=1)               # for multi-photo sessions
    is_primary = Column(Boolean, default=True)             # primary photo in a session
    
    # Parent reinspection link
    parent_inspection_id = Column(String(36), ForeignKey("inspections.id"), nullable=True)
    
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utcnow, index=True)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    
    batch = relationship("Batch", back_populates="inspections")
    operator = relationship("User", back_populates="inspections", foreign_keys=[operator_id])
    reviews = relationship("InspectionReview", back_populates="inspection")
    corrective_actions = relationship("CorrectiveAction", back_populates="inspection", foreign_keys="[CorrectiveAction.inspection_id]")
    child_inspections = relationship("Inspection", foreign_keys=[parent_inspection_id])
    
    __table_args__ = (
        Index("ix_inspections_batch_ts", "batch_id", "timestamp"),
        Index("ix_inspections_status", "inspection_status"),
    )


class InspectionReview(Base):
    __tablename__ = "inspection_reviews"
    
    id = Column(String(36), primary_key=True, default=new_uuid)
    inspection_id = Column(String(36), ForeignKey("inspections.id"), nullable=False, index=True)
    reviewer_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    # Confirmed outcome: GOOD|DEFECT|INCONCLUSIVE
    confirmed_class = Column(String(32), nullable=False)
    # Original model prediction preserved separately (not overwritten)
    original_prediction = Column(String(32), nullable=True)
    
    reason = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, default=utcnow)
    
    inspection = relationship("Inspection", back_populates="reviews")
    reviewer = relationship("User", back_populates="reviews")


class CorrectiveAction(Base):
    __tablename__ = "corrective_actions"
    
    id = Column(String(36), primary_key=True, default=new_uuid)
    inspection_id = Column(String(36), ForeignKey("inspections.id"), nullable=True)
    batch_id = Column(String(36), ForeignKey("batches.id"), nullable=True)
    
    title = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    root_cause = Column(Text, nullable=True)
    before_image_path = Column(String(512), nullable=True)
    after_image_path = Column(String(512), nullable=True)   # repair evidence
    
    assigned_to = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    
    due_date = Column(DateTime, nullable=True)
    
    # Status: open|in_progress|awaiting_reinspection|resolved|closed|reopened
    status = Column(String(32), default="open")
    priority = Column(String(16), default="medium")  # low|medium|high|critical
    
    completion_notes = Column(Text, nullable=True)
    follow_up_inspection_id = Column(String(36), ForeignKey("inspections.id"), nullable=True)
    reinspection_status = Column(String(32), default="pending")  # pending|passed|failed
    reinspection_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    closed_at = Column(DateTime, nullable=True)
    
    inspection = relationship("Inspection", back_populates="corrective_actions", foreign_keys=[inspection_id])
    assigned_user = relationship("User", back_populates="corrective_actions_assigned", foreign_keys=[assigned_to])
    created_by_user = relationship("User", back_populates="corrective_actions_created", foreign_keys=[created_by])


class AuditEvent(Base):
    __tablename__ = "audit_events"
    
    id = Column(String(36), primary_key=True, default=new_uuid)
    actor_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    actor_name = Column(String(128), nullable=True)
    action = Column(String(128), nullable=False)          # CREATE|UPDATE|DELETE|REVIEW|BATCH_STATUS_CHANGE|etc
    entity_type = Column(String(64), nullable=False)      # inspection|batch|corrective_action|user|etc
    entity_id = Column(String(36), nullable=True)
    previous_state = Column(Text, nullable=True)          # JSON
    new_state = Column(Text, nullable=True)               # JSON
    reason = Column(Text, nullable=True)
    ip_address = Column(String(64), nullable=True)
    timestamp = Column(DateTime, default=utcnow, index=True)


# ─── Database init ────────────────────────────────────────────────────────────

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    """FastAPI dependency for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hashlib.sha256(plain.encode()).hexdigest() == hashed


def seed_demo_data(db: Session):
    """Seed demo users, products, batches, inspections, reviews, CAPA, and audit events."""
    import random
    from datetime import timedelta
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    
    # 1. Demo users
    demo_users = [
        {"username": "admin", "display_name": "System Admin", "email": "admin@ps01demo.local",
         "password": "Admin@PS01Demo", "role": "admin"},
        {"username": "manager1", "display_name": "Quality Manager", "email": "manager@ps01demo.local",
         "password": "Manager@PS01Demo", "role": "manager"},
        {"username": "reviewer1", "display_name": "QC Reviewer", "email": "reviewer@ps01demo.local",
         "password": "Reviewer@PS01Demo", "role": "reviewer"},
        {"username": "operator1", "display_name": "Floor Operator A", "email": "operator1@ps01demo.local",
         "password": "Operator@PS01Demo", "role": "operator"},
        {"username": "operator2", "display_name": "Floor Operator B", "email": "operator2@ps01demo.local",
         "password": "Operator2@PS01Demo", "role": "operator"},
    ]
    
    created_users = {}
    for u in demo_users:
        existing = db.query(User).filter(User.username == u["username"]).first()
        if not existing:
            user = User(
                username=u["username"],
                display_name=u["display_name"],
                email=u["email"],
                password_hash=get_password_hash(u["password"]),
                role=u["role"],
            )
            db.add(user)
            db.flush()
            created_users[u["username"]] = user.id
        else:
            created_users[u["username"]] = existing.id
    
    manager_id = created_users.get("manager1")
    reviewer_id = created_users.get("reviewer1")
    op1_id = created_users.get("operator1")
    op2_id = created_users.get("operator2")
    
    # 2. Demo products
    demo_products = [
        {
            "name": "Steel Strip Type A",
            "product_type": "steel_strip",
            "sku": "SKU-STL-001",
            "inspection_category": "surface_defect",
            "inspection_criteria": "Surface scratch width < 0.2mm, no severe inclusions or scale pits",
            "threshold_defect": 0.45,
            "description": "Hot-rolled industrial steel strip, 256mm width",
        },
        {
            "name": "Steel Strip Type B",
            "product_type": "steel_strip",
            "sku": "SKU-STL-002",
            "inspection_category": "surface_defect",
            "inspection_criteria": "Cold-rolled mirror finish, zero allowable longitudinal gouges",
            "threshold_defect": 0.40,
            "description": "Cold-rolled steel strip, premium grade",
        },
        {
            "name": "Metal Casting Block",
            "product_type": "casting",
            "sku": "SKU-CST-003",
            "inspection_category": "casting_defect",
            "inspection_criteria": "Subsurface porosity diameter < 1.0mm, zero parting-line flash cracks",
            "threshold_defect": 0.50,
            "description": "Aluminium high-pressure die casting engine component",
        },
    ]
    
    created_products = {}
    for p in demo_products:
        existing = db.query(Product).filter(Product.name == p["name"]).first()
        if not existing:
            product = Product(**p)
            db.add(product)
            db.flush()
            created_products[p["name"]] = product.id
        else:
            created_products[p["name"]] = existing.id
            if not existing.sku:
                existing.sku = p["sku"]
                existing.inspection_criteria = p["inspection_criteria"]
    
    # 3. Demo batches
    demo_batches = [
        {
            "batch_code": "BATCH-2026-001",
            "product": "Steel Strip Type A",
            "quantity": 100,
            "production_line": "Line 1 - Hot Mill",
            "quality_status": "in_inspection",
            "notes": "Morning production shift A",
        },
        {
            "batch_code": "BATCH-2026-002",
            "product": "Steel Strip Type B",
            "quantity": 50,
            "production_line": "Line 2 - Cold Rolling",
            "quality_status": "on_hold",
            "hold_reason": "Severe surface pitting frequency exceeded 15% threshold during Shift 2.",
            "notes": "Quarantined pending supervisor review and roll calibration",
        },
        {
            "batch_code": "BATCH-2026-003",
            "product": "Metal Casting Block",
            "quantity": 25,
            "production_line": "Line 3 - Die Casting Foundry",
            "quality_status": "in_inspection",
            "notes": "Pressure testing underway",
        },
        {
            "batch_code": "BATCH-2026-004",
            "product": "Steel Strip Type A",
            "quantity": 200,
            "production_line": "Line 1 - Hot Mill",
            "quality_status": "approved",
            "notes": "Quality release approved by Manager following 100% compliant sampling",
        },
    ]
    
    created_batches = {}
    for b in demo_batches:
        existing = db.query(Batch).filter(Batch.batch_code == b["batch_code"]).first()
        if not existing:
            product_id = created_products.get(b["product"])
            if product_id:
                batch = Batch(
                    batch_code=b["batch_code"],
                    product_id=product_id,
                    quantity=b["quantity"],
                    production_line=b["production_line"],
                    quality_status=b["quality_status"],
                    hold_reason=b.get("hold_reason"),
                    notes=b.get("notes"),
                    created_by=manager_id,
                )
                db.add(batch)
                db.flush()
                created_batches[b["batch_code"]] = batch.id
        else:
            created_batches[b["batch_code"]] = existing.id
            if b.get("hold_reason") and not existing.hold_reason:
                existing.hold_reason = b["hold_reason"]
                existing.quality_status = b["quality_status"]
    
    # 4. Demo inspections (using real sample images)
    demo_inspections = [
        {
            "id": "INSP-2026-001",
            "batch_code": "BATCH-2026-001",
            "operator_id": op1_id,
            "serial_number": "ST-001-A",
            "prediction": "GOOD",
            "defect_score": 0.08,
            "good_score": 0.92,
            "image_filename": "000789191.jpg",
            "image_path": "images/000789191.jpg",
            "inspection_status": "completed",
            "notes": "Clean smooth surface, passed specification",
        },
        {
            "id": "INSP-2026-002",
            "batch_code": "BATCH-2026-001",
            "operator_id": op1_id,
            "serial_number": "ST-001-B",
            "prediction": "DEFECT",
            "defect_score": 0.92,
            "good_score": 0.08,
            "image_filename": "0007a71bf.jpg",
            "image_path": "images/0007a71bf.jpg",
            "inspection_status": "under_review",
            "notes": "Longitudinal scratch along central axis",
        },
        {
            "id": "INSP-2026-003",
            "batch_code": "BATCH-2026-002",
            "operator_id": op2_id,
            "serial_number": "ST-002-C",
            "prediction": "DEFECT",
            "defect_score": 0.88,
            "good_score": 0.12,
            "image_filename": "001982b08.jpg",
            "image_path": "images/001982b08.jpg",
            "inspection_status": "reviewed",
            "notes": "Repetitive pit inclusions near left margin",
        },
        {
            "id": "INSP-2026-004",
            "batch_code": "BATCH-2026-002",
            "operator_id": op2_id,
            "serial_number": "ST-002-D",
            "prediction": "UNCERTAIN",
            "defect_score": 0.42,
            "good_score": 0.58,
            "image_filename": "cast_def_0_1059.jpeg",
            "image_path": "images/cast_def_0_1059.jpeg",
            "inspection_status": "under_review",
            "notes": "Faint boundary shade, confidence near threshold",
        },
        {
            "id": "INSP-2026-005",
            "batch_code": "BATCH-2026-003",
            "operator_id": op1_id,
            "serial_number": "CST-003-E",
            "prediction": "DEFECT",
            "defect_score": 0.89,
            "good_score": 0.11,
            "image_filename": "cast_def_0_1063.jpeg",
            "image_path": "images/cast_def_0_1063.jpeg",
            "inspection_status": "reviewed",
            "notes": "Surface blowhole cavity detected near boss",
        },
        {
            "id": "INSP-2026-006",
            "batch_code": "BATCH-2026-003",
            "operator_id": op1_id,
            "serial_number": "CST-003-E-RE",
            "prediction": "GOOD",
            "defect_score": 0.06,
            "good_score": 0.94,
            "image_filename": "cast_def_0_108.jpeg",
            "image_path": "images/cast_def_0_108.jpeg",
            "inspection_status": "completed",
            "parent_inspection_id": "INSP-2026-005",
            "notes": "Post-repair reinspection after weld fill & grind. Passed zero porosity test.",
        },
    ]
    
    for insp_data in demo_inspections:
        existing = db.query(Inspection).filter(Inspection.id == insp_data["id"]).first()
        batch_id = created_batches.get(insp_data["batch_code"])
        if not existing and batch_id:
            insp = Inspection(
                id=insp_data["id"],
                batch_id=batch_id,
                operator_id=insp_data["operator_id"],
                serial_number=insp_data["serial_number"],
                capture_mode="photo",
                image_filename=insp_data["image_filename"],
                image_path=insp_data["image_path"],
                prediction=insp_data["prediction"],
                defect_score=insp_data["defect_score"],
                good_score=insp_data["good_score"],
                model_version="1.0.0",
                inference_time_ms=48.5,
                raw_scores=json.dumps({"GOOD": insp_data["good_score"], "DEFECT": insp_data["defect_score"]}),
                inspection_status=insp_data["inspection_status"],
                session_id=f"sess-{insp_data['id']}",
                parent_inspection_id=insp_data.get("parent_inspection_id"),
                notes=insp_data["notes"],
                timestamp=now - timedelta(hours=random.randint(1, 8)),
            )
            db.add(insp)
    db.flush()
    
    # 5. Demo reviews
    reviews_data = [
        {
            "inspection_id": "INSP-2026-003",
            "reviewer_id": reviewer_id,
            "confirmed_class": "DEFECT",
            "original_prediction": "DEFECT",
            "reason": "Confirmed severe pitting and roller score mark",
            "notes": "Triggered root-cause investigation for Line 2",
        },
        {
            "inspection_id": "INSP-2026-005",
            "reviewer_id": reviewer_id,
            "confirmed_class": "DEFECT",
            "original_prediction": "DEFECT",
            "reason": "Confirmed casting blowhole cavity exceeding 1mm limit",
            "notes": "Sent for precision weld rework and reinspection",
        },
    ]
    for r in reviews_data:
        existing = db.query(InspectionReview).filter(InspectionReview.inspection_id == r["inspection_id"]).first()
        if not existing:
            rev = InspectionReview(**r, reviewed_at=now - timedelta(hours=2))
            db.add(rev)
    db.flush()
    
    # 6. Demo corrective actions (including Overdue and Reinspection Resolved)
    ca_data = [
        {
            "id": "CAPA-2026-001",
            "inspection_id": "INSP-2026-003",
            "batch_id": created_batches.get("BATCH-2026-002"),
            "title": "Line 2 Guide roller dressing and bearing alignment",
            "description": "Repetitive scoring marks caused by chipped roll edge. Regrind rollers and replace bearing assembly.",
            "root_cause": "Roller bearing vibration and foreign particulate abrasion on pinch rollers",
            "assigned_to": reviewer_id,
            "created_by": reviewer_id,
            "priority": "high",
            "status": "in_progress",
            "due_date": now + timedelta(days=2),
            "before_image_path": "images/001982b08.jpg",
        },
        {
            "id": "CAPA-2026-002",
            "inspection_id": "INSP-2026-002",
            "batch_id": created_batches.get("BATCH-2026-001"),
            "title": "Coolant filtration system flush and scale nozzle replacement",
            "description": "High temperature oxidation scale forming longitudinal gouges. Flush coolant circuit and verify flow rate.",
            "root_cause": "Clogged pressure nozzles causing uneven cooling and thermal fissures",
            "assigned_to": op2_id,
            "created_by": manager_id,
            "priority": "critical",
            "status": "open",
            "due_date": now - timedelta(days=2),  # OVERDUE!
            "before_image_path": "images/0007a71bf.jpg",
        },
        {
            "id": "CAPA-2026-003",
            "inspection_id": "INSP-2026-005",
            "batch_id": created_batches.get("BATCH-2026-003"),
            "title": "Foundry mold release agent recalibration and weld repair",
            "description": "Porosity blowholes repaired via TIG fill and surface linishing. Recalibrate automated lubricant spray.",
            "root_cause": "Excess moisture in mold release agent spray generating gas entrapment",
            "assigned_to": op1_id,
            "created_by": reviewer_id,
            "priority": "medium",
            "status": "resolved",
            "due_date": now - timedelta(days=1),
            "before_image_path": "images/cast_def_0_1063.jpeg",
            "after_image_path": "images/cast_def_0_108.jpeg",
            "reinspection_status": "passed",
            "follow_up_inspection_id": "INSP-2026-006",
            "completion_notes": "Spray volume adjusted to 2.4L/min. Follow-up inspection INSP-2026-006 confirmed zero porosity.",
        },
    ]
    for c in ca_data:
        existing = db.query(CorrectiveAction).filter(CorrectiveAction.id == c["id"]).first()
        if not existing:
            action = CorrectiveAction(**c)
            db.add(action)
    db.flush()
    
    # 7. Demo audit events
    audit_data = [
        {"actor_name": "Quality Manager", "action": "CREATE", "entity_type": "batch", "entity_id": created_batches.get("BATCH-2026-001"), "new_state": json.dumps({"batch_code": "BATCH-2026-001", "status": "in_inspection"})},
        {"actor_name": "Quality Manager", "action": "BATCH_STATUS_CHANGE", "entity_type": "batch", "entity_id": created_batches.get("BATCH-2026-002"), "previous_state": json.dumps({"quality_status": "in_inspection"}), "new_state": json.dumps({"quality_status": "on_hold"}), "reason": "Severe surface pitting frequency exceeded 15% threshold."},
        {"actor_name": "QC Reviewer", "action": "REVIEW", "entity_type": "inspection", "entity_id": "INSP-2026-003", "new_state": json.dumps({"confirmed_class": "DEFECT"}), "reason": "Confirmed severe pitting and roller score mark."},
        {"actor_name": "QC Reviewer", "action": "CREATE", "entity_type": "corrective_action", "entity_id": "CAPA-2026-001", "new_state": json.dumps({"title": "Line 2 Guide roller dressing"})},
        {"actor_name": "Quality Manager", "action": "BATCH_STATUS_CHANGE", "entity_type": "batch", "entity_id": created_batches.get("BATCH-2026-004"), "previous_state": json.dumps({"quality_status": "in_inspection"}), "new_state": json.dumps({"quality_status": "approved"}), "reason": "Batch released following 100% compliant sampling."},
    ]
    for a in audit_data:
        existing = db.query(AuditEvent).filter(AuditEvent.entity_id == a["entity_id"], AuditEvent.action == a["action"]).first()
        if not existing:
            evt = AuditEvent(**a, timestamp=now - timedelta(hours=random.randint(1, 12)))
            db.add(evt)
    
    db.commit()
    print("[DB] Industrial demo records seeded successfully.")
    print("[DB] Demo accounts:")
    for u in demo_users:
        print(f"       {u['role']:10s}: {u['username']} / {u['password']}")


def init_db():
    """Create tables and seed demo data."""
    create_tables()
    db = SessionLocal()
    try:
        seed_demo_data(db)
    finally:
        db.close()
