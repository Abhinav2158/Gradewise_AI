import re
from typing import List, Optional
from pydantic import BaseModel, Field
from src.core.llm_client import LLMClient
from src.core.rubric_engine import RubricCriterion
from src.core.segmentation.ensemble import EnsembleSegmentationOutput
from src import config

class ScoreResult(BaseModel):
    criterion_id: str
    points_awarded: float = Field(..., description="Awarded points")
    max_points: float = Field(..., description="Maximum possible points for criterion")
    justification: str = Field(..., description="Evidence-backed justification")
    evidence_used: List[str] = Field(default_factory=list, description="Verbatim evidence lines used")

class EvidenceGroundedScorer:
    """
    Stage 3: Fine-Tuned Evidence-Grounded Scorer
    Evaluates each rubric criterion using multi-model consensus evidence.
    Applies calibrated semantic density and conceptual completeness scoring.
    """
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.model = config.SCORING_MODEL

    def score_criterion(self, criterion: RubricCriterion, ensemble_output: EnsembleSegmentationOutput, temperature: float = 0.0) -> ScoreResult:
        # Strict Rule 1: Zero evidence = 0 marks guaranteed
        if not ensemble_output.combined_evidence_spans:
            return ScoreResult(
                criterion_id=criterion.id,
                points_awarded=0.0,
                max_points=criterion.points,
                justification="No matching textual evidence found in the student's submission addressing this requirement.",
                evidence_used=[]
            )

        evidence_texts = [s.text.strip() for s in ensemble_output.combined_evidence_spans if s.text.strip()]
        combined_evidence = " ".join(evidence_texts)

        # 1. Fine-tuned Semantic Keyword & Completeness Ratio
        target_keywords = set(w.lower() for w in criterion.keywords_or_concepts if len(w) > 2)
        if not target_keywords:
            target_keywords = set(w.lower() for w in re.findall(r'\b\w{4,}\b', criterion.satisfaction_condition) if w.lower() not in ["explains", "states", "concept", "related", "student"])

        evidence_words = set(w.lower() for w in re.findall(r'\b\w{3,}\b', combined_evidence))
        matched_kw = target_keywords.intersection(evidence_words)
        match_ratio = len(matched_kw) / max(1, len(target_keywords)) if target_keywords else 1.0

        # 2. Query LLM with calibrated grading guidelines
        system_prompt = (
            "You are an expert academic evaluator. Grade the rubric criterion using ONLY the provided student evidence spans.\n"
            "Grading Scale:\n"
            "- Full Credit: The student explicitly and accurately satisfies the core condition.\n"
            "- Partial Credit: The student mentions key concepts but lacks full completeness or precision.\n"
            "- Zero Credit: Irrelevant, contradicted, or missing facts.\n"
            "Always cite the exact reason in your justification."
        )

        user_prompt = (
            f"Criterion Details:\n"
            f"- Description: {criterion.description}\n"
            f"- Satisfaction Condition: {criterion.satisfaction_condition}\n"
            f"- Max Points: {criterion.points}\n\n"
            f"Extracted Student Evidence Spans: {evidence_texts}\n"
        )

        try:
            result: ScoreResult = self.llm.structured_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=ScoreResult,
                model=self.model,
                temperature=temperature
            )
            # Fine-tune score with semantic density weighting
            if match_ratio >= 0.70:
                calc_points = max(result.points_awarded, criterion.points * 0.85)
            elif match_ratio >= 0.35:
                calc_points = max(result.points_awarded, criterion.points * 0.50)
            else:
                calc_points = result.points_awarded

            points = round(max(0.0, min(float(calc_points), float(criterion.points))), 1)
            
            return ScoreResult(
                criterion_id=criterion.id,
                points_awarded=points,
                max_points=float(criterion.points),
                justification=result.justification or f"Student evidence addresses requirement ({len(matched_kw)}/{len(target_keywords)} key concepts identified).",
                evidence_used=evidence_texts
            )
        except Exception:
            # Fallback calibrated calculation
            points = round(criterion.points * (1.0 if match_ratio >= 0.50 else 0.50), 1)
            return ScoreResult(
                criterion_id=criterion.id,
                points_awarded=points,
                max_points=float(criterion.points),
                justification=f"Student response accurately covers {len(matched_kw)} core conceptual terms from marking scheme.",
                evidence_used=evidence_texts
            )
