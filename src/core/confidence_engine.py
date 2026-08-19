import numpy as np
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from src.core.rubric_engine import RubricCriterion, RubricSchema
from src.core.segmentation.ensemble import SegmentationEnsemble, EnsembleSegmentationOutput
from src.core.scorer import EvidenceGroundedScorer, ScoreResult
from src import config

class CriterionGradingResult(BaseModel):
    criterion: RubricCriterion
    segmentation: EnsembleSegmentationOutput
    score_result: ScoreResult
    confidence_score: float
    routing: str  # "auto_accept", "flag_for_spot_check", "requires_review"

class AnswerGradingReport(BaseModel):
    student_id: int
    question_id: str
    total_score: float
    max_total_score: float
    composite_confidence: float
    overall_routing: str
    criterion_results: List[CriterionGradingResult]

class ConfidenceEngine:
    """
    Stage 4: Confidence Scoring Engine
    Calculates multi-signal confidence from:
    1. Cross-backend span overlap (Jaccard)
    2. Score variance (self-consistency check)
    3. Evidence unanimity across 3 backends
    """
    def __init__(self, ensemble: SegmentationEnsemble, scorer: EvidenceGroundedScorer):
        self.ensemble = ensemble
        self.scorer = scorer

    def evaluate_criterion(self, criterion: RubricCriterion, student_answer: str) -> CriterionGradingResult:
        # Step 1: Run multi-model segmentation ensemble
        seg_output = self.ensemble.segment_answer(criterion, student_answer)

        # Step 2: Run primary scoring (T=0)
        score_1 = self.scorer.score_criterion(criterion, seg_output, temperature=0.0)

        # Step 3: Run secondary scoring pass for variance check (T=0.3)
        score_2 = self.scorer.score_criterion(criterion, seg_output, temperature=0.3)

        # Compute normalized score variance
        max_p = max(1.0, float(criterion.points))
        score_diff = abs(score_1.points_awarded - score_2.points_awarded) / max_p
        score_variance_normalized = min(1.0, score_diff)

        # Cross-backend span overlap agreement
        span_overlap = seg_output.span_overlap_agreement

        # Evidence agreement (1.0 if unanimous, 0.5 if majority, 0.0 if divided)
        if seg_output.evidence_found_unanimous:
            evidence_agreement = 1.0
        elif seg_output.evidence_found_majority:
            evidence_agreement = 0.65
        else:
            evidence_agreement = 0.2

        # Weighted composite confidence formula
        confidence = (
            0.40 * span_overlap +
            0.40 * (1.0 - score_variance_normalized) +
            0.20 * evidence_agreement
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Routing Decision
        if confidence >= config.CONFIDENCE_AUTO_ACCEPT:
            route = "auto_accept"
        elif confidence >= config.CONFIDENCE_SPOT_CHECK:
            route = "flag_for_spot_check"
        else:
            route = "requires_review"

        return CriterionGradingResult(
            criterion=criterion,
            segmentation=seg_output,
            score_result=score_1,
            confidence_score=confidence,
            routing=route
        )

    def grade_full_answer(self, student_id: int, rubric: RubricSchema, student_answer: str) -> AnswerGradingReport:
        """Grades all criteria for a student answer and outputs the full report."""
        criterion_results = []
        total_score = 0.0
        max_total_score = 0.0
        conf_scores = []

        for crit in rubric.criteria:
            res = self.evaluate_criterion(crit, student_answer)
            criterion_results.append(res)
            total_score += res.score_result.points_awarded
            max_total_score += crit.points
            conf_scores.append(res.confidence_score)

        avg_confidence = float(np.mean(conf_scores)) if conf_scores else 0.0

        # Answer routing is conservative: if any criterion requires review, route to review
        has_review = any(c.routing == "requires_review" for c in criterion_results)
        has_spot_check = any(c.routing == "flag_for_spot_check" for c in criterion_results)

        if has_review:
            overall_routing = "requires_review"
        elif has_spot_check:
            overall_routing = "flag_for_spot_check"
        else:
            overall_routing = "auto_accept"

        return AnswerGradingReport(
            student_id=student_id,
            question_id=rubric.question_id,
            total_score=round(total_score, 2),
            max_total_score=round(max_total_score, 2),
            composite_confidence=round(avg_confidence, 3),
            overall_routing=overall_routing,
            criterion_results=criterion_results
        )
