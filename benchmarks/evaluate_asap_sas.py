import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR
from src.core.llm_client import LLMClient
from src.core.rubric_engine import RubricEngine, RubricSchema
from src.core.segmentation.llm_backend import LLMSegmentationBackend
from src.core.segmentation.ensemble import SegmentationEnsemble
from src.core.scorer import EvidenceGroundedScorer
from src.core.confidence_engine import ConfidenceEngine
from src.utils.metrics import compute_qwk

def run_asap_benchmark(set_id: str = "1"):
    dataset_path = DATA_DIR / "asap_sas" / f"asap_set_{set_id}.json"
    if not dataset_path.exists():
        print(f"Dataset file not found: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    question_id = data["question_id"]
    question_text = data["question_text"]
    reference_answer = data["reference_answer"]
    total_marks = data["total_marks"]
    records = data["sample_records"]

    print(f"\n=======================================================")
    print(f"BENCHMARK EVALUATION: ASAP-SAS (Prompt Set {set_id})")
    print(f"Question: {question_text[:70]}...")
    print(f"Evaluating {len(records)} authentic student responses...")
    print(f"=======================================================\n")

    # Initialize Pipeline Components
    llm = LLMClient()
    rubric_engine = RubricEngine(llm)
    
    # Stage 1: Rubric Generation
    print("-> Stage 1: Generating atomic rubric criteria...")
    rubric: RubricSchema = rubric_engine.generate_rubric(
        question_id=question_id,
        question_text=question_text,
        reference_answer=reference_answer,
        total_marks=total_marks
    )
    print(f"Generated {len(rubric.criteria)} atomic criteria (Total Marks: {rubric.total_marks}):")
    for c in rubric.criteria:
        print(f"   [{c.id}] ({c.points} pts): {c.description}")

    # Initialize Ensemble & Scoring Pipeline
    ensemble = SegmentationEnsemble(LLMSegmentationBackend(llm))
    scorer = EvidenceGroundedScorer(llm)
    conf_engine = ConfidenceEngine(ensemble, scorer)

    ai_scores = []
    h1_scores = []
    h2_scores = []
    routings = []
    confidences = []

    print("\n-> Executing Stages 2, 3 & 4 (Segmentation Ensemble, Scoring, Confidence Gating)...")
    for rec in records:
        sid = rec["id"]
        ans = rec["student_answer"]
        h1 = int(rec["human_score_1"])
        h2 = int(rec["human_score_2"])

        report = conf_engine.grade_full_answer(student_id=sid, rubric=rubric, student_answer=ans)
        
        # Round AI score to integer for QWK calculation
        ai_score_int = int(round(report.total_score))

        ai_scores.append(ai_score_int)
        h1_scores.append(h1)
        h2_scores.append(h2)
        routings.append(report.overall_routing)
        confidences.append(report.composite_confidence)

        print(f"  Student #{sid} | Human1: {h1} | Human2: {h2} | AI: {ai_score_int} (raw: {report.total_score:.1f}) | Conf: {report.composite_confidence:.2f} | Route: {report.overall_routing}")

    # Compute Benchmark Metrics
    qwk_ai_vs_h1 = compute_qwk(h1_scores, ai_scores)
    qwk_ai_vs_h2 = compute_qwk(h2_scores, ai_scores)
    qwk_h1_vs_h2 = compute_qwk(h1_scores, h2_scores)

    auto_accept_pct = (routings.count("auto_accept") / len(routings)) * 100
    spot_check_pct = (routings.count("flag_for_spot_check") / len(routings)) * 100
    review_pct = (routings.count("requires_review") / len(routings)) * 100

    print("\n=======================================================")
    print("             FINAL BENCHMARK RESULTS                   ")
    print("=======================================================")
    print(f" Metric: Quadratic Weighted Kappa (QWK)")
    print(f" - System vs Human Grader 1:        {qwk_ai_vs_h1:.4f}")
    print(f" - System vs Human Grader 2:        {qwk_ai_vs_h2:.4f}")
    print(f" - Human 1 vs Human 2 (Ceiling):    {qwk_h1_vs_h2:.4f}")
    print(f"-------------------------------------------------------")
    print(f" Confidence Routing Distribution:")
    print(f" - Auto-Accepted (High Conf >= 0.80):  {auto_accept_pct:.1f}%")
    print(f" - Flagged Spot-Check (0.50 - 0.79):   {spot_check_pct:.1f}%")
    print(f" - Required Human Review (< 0.50):    {review_pct:.1f}%")
    print("=======================================================\n")

if __name__ == "__main__":
    run_asap_benchmark(set_id="1")
    run_asap_benchmark(set_id="2")
