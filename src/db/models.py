import json
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class RubricVersion(Base):
    __tablename__ = "rubric_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(String(64), nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)
    schema_json = Column(Text, nullable=False) # JSON of RubricSchema
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    grading_records = relationship("GradingRecord", back_populates="rubric_version")

class StudentSubmission(Base):
    __tablename__ = "student_submissions"

    id = Column(Integer, primary_key=True)
    question_id = Column(String(64), nullable=False, index=True)
    student_id = Column(Integer, nullable=False)
    answer_text = Column(Text, nullable=False)
    human_score_1 = Column(Float, nullable=True)
    human_score_2 = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    grading_records = relationship("GradingRecord", back_populates="submission")

class GradingRecord(Base):
    __tablename__ = "grading_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    submission_id = Column(Integer, ForeignKey("student_submissions.id"), nullable=False)
    rubric_version_id = Column(Integer, ForeignKey("rubric_versions.id"), nullable=False)
    criterion_id = Column(String(64), nullable=False)
    
    # AI Grading Details
    evidence_spans_json = Column(Text, nullable=False) # JSON of extracted spans
    tentative_score = Column(Float, nullable=False)
    max_points = Column(Float, nullable=False)
    justification = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    routing_decision = Column(String(32), nullable=False) # "auto_accept", "flag_for_spot_check", "requires_review"

    # Human Override Details (Stage 5)
    is_overridden = Column(Boolean, default=False)
    final_score = Column(Float, nullable=False)
    instructor_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    submission = relationship("StudentSubmission", back_populates="grading_records")
    rubric_version = relationship("RubricVersion", back_populates="grading_records")

class AuditTrail(Base):
    __tablename__ = "audit_trails"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action_type = Column(String(64), nullable=False) # "RUBRIC_AMENDMENT", "SCORE_OVERRIDE", "RE_SCORE_BATCH"
    details_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
