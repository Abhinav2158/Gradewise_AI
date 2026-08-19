import json
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.db.models import RubricVersion, StudentSubmission, GradingRecord, AuditTrail
from src.core.rubric_engine import RubricSchema, RubricRefinementResult
from src.core.confidence_engine import ConfidenceEngine

class RescoringQueue:
    """
    Stage 7: Event-driven Retroactive Re-scoring Queue
    When a rubric is updated/refined, finds all historical submissions affected by the
    amended or newly added criteria and triggers automated re-evaluation.
    """
    def __init__(self, db: Session, confidence_engine: ConfidenceEngine):
        self.db = db
        self.engine = confidence_engine

    def apply_rubric_refinement(self, question_id: str, new_rubric_schema: RubricSchema, refinement_result: RubricRefinementResult) -> Dict[str, Any]:
        # 1. Fetch current active rubric version
        active_version = self.db.query(RubricVersion).filter(
            RubricVersion.question_id == question_id,
            RubricVersion.is_active == True
        ).first()

        new_version_num = (active_version.version + 1) if active_version else 1

        if active_version:
            active_version.is_active = False

        # 2. Save new rubric version
        new_version_record = RubricVersion(
            question_id=question_id,
            version=new_version_num,
            schema_json=new_rubric_schema.model_dump_json(),
            is_active=True,
            notes=refinement_result.rationale
        )
        self.db.add(new_version_record)
        self.db.commit()
        self.db.refresh(new_version_record)

        # 3. Find all student submissions for this question
        submissions = self.db.query(StudentSubmission).filter(
            StudentSubmission.question_id == question_id
        ).all()

        rescored_count = 0
        re_scored_details = []

        # 4. Re-evaluate each submission with the new rubric
        for sub in submissions:
            # Check previous score
            prev_records = self.db.query(GradingRecord).filter(
                GradingRecord.submission_id == sub.id,
                GradingRecord.rubric_version_id == (active_version.id if active_version else None)
            ).all()
            prev_total = sum(r.final_score for r in prev_records) if prev_records else 0.0

            report = self.engine.grade_full_answer(
                student_id=sub.student_id,
                rubric=new_rubric_schema,
                student_answer=sub.answer_text
            )

            # Monotonic Grace Rule: Automated rescoring can only increase or preserve published scores
            is_quarantined = False
            effective_final_score = report.total_score
            if prev_records and report.total_score < prev_total:
                is_quarantined = True
                effective_final_score = prev_total  # Preserve previous published grade

            # Record updated grading records
            for crit_res in report.criterion_results:
                spans_json = json.dumps([s.model_dump() for s in crit_res.segmentation.combined_evidence_spans])
                g_record = GradingRecord(
                    submission_id=sub.id,
                    rubric_version_id=new_version_record.id,
                    criterion_id=crit_res.criterion.id,
                    evidence_spans_json=spans_json,
                    tentative_score=crit_res.score_result.points_awarded,
                    max_points=crit_res.criterion.points,
                    justification=crit_res.score_result.justification,
                    confidence_score=crit_res.confidence_score,
                    routing_decision="instructor_manual_override_required" if is_quarantined else crit_res.routing,
                    final_score=crit_res.score_result.points_awarded
                )
                self.db.add(g_record)

            rescored_count += 1
            re_scored_details.append({
                "student_id": sub.student_id,
                "previous_total": prev_total,
                "new_total_score": effective_final_score,
                "is_quarantined_lowering": is_quarantined,
                "confidence": report.composite_confidence,
                "routing": "instructor_manual_override_required" if is_quarantined else report.overall_routing
            })

        # 5. Log audit trail
        audit = AuditTrail(
            action_type="RUBRIC_AMENDMENT_RE_SCORE",
            details_json=json.dumps({
                "question_id": question_id,
                "from_version": active_version.version if active_version else 0,
                "to_version": new_version_num,
                "rescored_answers_count": rescored_count,
                "monotonic_grace_enforced": True,
                "refinement_rationale": refinement_result.rationale
            })
        )
        self.db.add(audit)
        self.db.commit()

        return {
            "new_version": new_version_num,
            "rescored_count": rescored_count,
            "details": re_scored_details
        }
