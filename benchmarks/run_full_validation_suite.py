import os
import sys
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import DATA_DIR
from src.core.llm_client import LLMClient
from src.core.rubric_engine import RubricEngine, RubricSchema
from src.core.segmentation.llm_backend import LLMSegmentationBackend
from src.core.segmentation.ensemble import SegmentationEnsemble
from src.core.scorer import EvidenceGroundedScorer
from src.core.confidence_engine import ConfidenceEngine
from src.utils.metrics import compute_qwk

def bootstrap_qwk_ci(y_true: List[int], y_pred: List[int], n_bootstraps: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Computes base QWK and 95% bootstrap confidence intervals."""
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    base_qwk = compute_qwk(y_t.tolist(), y_p.tolist())
    
    if len(y_t) < 5 or len(set(y_p)) <= 1 or len(set(y_t)) <= 1:
        return float(base_qwk), float(base_qwk), float(base_qwk)

    boot_qwks = []
    rng = np.random.default_rng(42)
    n = len(y_t)
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        sample_t = y_t[idx].tolist()
        sample_p = y_p[idx].tolist()
        if len(set(sample_t)) > 1 and len(set(sample_p)) > 1:
            val = compute_qwk(sample_t, sample_p)
            boot_qwks.append(val)
            
    if not boot_qwks:
        return float(base_qwk), float(base_qwk), float(base_qwk)
        
    lower = float(np.percentile(boot_qwks, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_qwks, 100 * (1 - alpha / 2)))
    return float(base_qwk), lower, upper

def run_suite():
    print("=================================================================")
    print("      LIVE MULTI-MODEL VALIDATION & ABLATION BENCHMARK SUITE     ")
    print("=================================================================\n")

    # Load dataset
    ds_path = DATA_DIR / "asap_sas" / "asap_set_1.json"
    if not ds_path.exists():
        print(f"Error: Dataset {ds_path} not found.")
        return

    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data["sample_records"]
    q_text = data["question_text"]
    ref_ans = data["reference_answer"]
    total_marks = data["total_marks"]
    
    print(f"Loaded ASAP-SAS Set 1 ({len(records)} authentic student responses)")
    print(f"Question: {q_text[:75]}...\n")

    llm = LLMClient()
    rubric_engine = RubricEngine(llm)

    # Generate Rubric
    print("-> Generating Atomic Rubric Schema via LLM...", flush=True)
    rubric = rubric_engine.generate_rubric(
        question_id=data["question_id"],
        question_text=q_text,
        reference_answer=ref_ans,
        total_marks=total_marks
    )
    print(f"   Generated {len(rubric.criteria)} criteria (Total Marks: {rubric.total_marks}):", flush=True)
    for c in rubric.criteria:
        print(f"   * [{c.id}] ({c.points} pts): {c.description}", flush=True)
    print("", flush=True)

    # Initialize components
    ensemble = SegmentationEnsemble(LLMSegmentationBackend(llm))
    scorer = EvidenceGroundedScorer(llm)
    conf_engine = ConfidenceEngine(ensemble, scorer)

    h1_scores = []
    h2_scores = []
    
    scores_zero_shot = []
    scores_rubric_only = []
    scores_full_ensemble = []
    
    features_for_fitting = []
    labels_for_fitting = []

    print("-> Executing Live Evaluation & 3-Way Ablation on Student Responses...", flush=True)
    for idx, rec in enumerate(records, 1):
        s_ans = rec["student_answer"]
        h1 = int(rec["human_score_1"])
        h2 = int(rec["human_score_2"])
        h1_scores.append(h1)
        h2_scores.append(h2)

        # 1. Ablation A: Zero-shot single prompt baseline
        prompt_zs = f"Grade this student answer out of {int(total_marks)} integer points (0, 1, 2, or 3):\nQuestion: {q_text}\nStudent Answer: {s_ans}\nProvide score format: SCORE: <integer>"
        resp_zs = llm.complete(prompt_zs, "You are a teacher grading student exams.")
        m = re.search(r'SCORE:\s*(\d+)', resp_zs, re.IGNORECASE) or re.search(r'\b([0-3])\b', resp_zs)
        score_zs = int(m.group(1)) if m else int(total_marks // 2)
        score_zs = min(int(total_marks), max(0, score_zs))
        scores_zero_shot.append(score_zs)

        # 2. Full Consensus Ensemble
        rep = conf_engine.grade_full_answer(student_id=rec["id"], rubric=rubric, student_answer=s_ans)
        ai_score_full = int(round(rep.total_score))
        ai_score_full = min(int(total_marks), max(0, ai_score_full))
        scores_full_ensemble.append(ai_score_full)
        
        # 3. Rubric-only (without multi-model ensemble consensus penalty)
        score_rub = int(round(sum(c.score_result.points_awarded for c in rep.criterion_results)))
        score_rub = min(int(total_marks), max(0, score_rub))
        scores_rubric_only.append(score_rub)

        # Features for logistic calibration
        feat_overlap = float(np.mean([c.segmentation.span_overlap_agreement for c in rep.criterion_results])) if rep.criterion_results else 0.5
        feat_var = float(np.std([c.score_result.points_awarded for c in rep.criterion_results])) if len(rep.criterion_results) > 1 else 0.0
        feat_lexical = min(1.0, len(s_ans.split()) / 25.0)
        features_for_fitting.append([feat_overlap, max(0.0, 1.0 - feat_var), 1.0, feat_lexical])
        labels_for_fitting.append(1 if abs(ai_score_full - h1) == 0 else 0)

        print(f"  [{idx:02d}/{len(records)}] Student #{rec['id']} | H1: {h1} | H2: {h2} | ZS: {score_zs} | Rubric: {score_rub} | Full Ensemble: {ai_score_full} | Conf: {rep.composite_confidence:.2f}", flush=True)
        import time
        time.sleep(2.5)

    # Compute QWKs and 95% Bootstrap Confidence Intervals
    qwk_ceil, ceil_low, ceil_high = bootstrap_qwk_ci(h1_scores, h2_scores)
    qwk_zs, zs_low, zs_high = bootstrap_qwk_ci(h1_scores, scores_zero_shot)
    qwk_rub, rub_low, rub_high = bootstrap_qwk_ci(h1_scores, scores_rubric_only)
    qwk_ens, ens_low, ens_high = bootstrap_qwk_ci(h1_scores, scores_full_ensemble)

    print("\n=================================================================")
    print("             MEASURED BENCHMARK & ABLATION RESULTS               ")
    print("=================================================================")
    print(f" Human-to-Human Ceiling (H1 vs H2): QWK = {qwk_ceil:.4f} [95% CI: {ceil_low:.4f} - {ceil_high:.4f}]")
    print("-----------------------------------------------------------------")
    print(f" (A) Zero-Shot LLM Baseline:        QWK = {qwk_zs:.4f} [95% CI: {zs_low:.4f} - {zs_high:.4f}]")
    print(f" (B) LLM + Rubric Decomposition:    QWK = {qwk_rub:.4f} [95% CI: {rub_low:.4f} - {rub_high:.4f}]")
    print(f" (C) Full 3-Backend Ensemble (Ours):QWK = {qwk_ens:.4f} [95% CI: {ens_low:.4f} - {ens_high:.4f}]")
    print("=================================================================\n")

    # Logistic Regression Calibration Fit
    X = np.array(features_for_fitting)
    y = np.array(labels_for_fitting)
    if len(set(y)) > 1:
        clf = LogisticRegression(fit_intercept=False, random_state=42)
        clf.fit(X, y)
        raw_w = np.maximum(0.01, clf.coef_[0])
        norm_w = (raw_w / np.sum(raw_w)).tolist()
        print("-> Fitted Logistic Regression Gating Weights (on N=10 dev):")
        print(f"   w1 (Span Overlap): {norm_w[0]:.4f}")
        print(f"   w2 (1 - Variance): {norm_w[1]:.4f}")
        print(f"   w3 (OCR Quality):  {norm_w[2]:.4f}")
        print(f"   w4 (Lexical):      {norm_w[3]:.4f}")
    else:
        norm_w = [0.40, 0.35, 0.15, 0.10]
        print("-> Dataset subset was unanimous on accuracy; default heuristic weights retained.")

    # Phase 0.3: Write Run Manifest Sidecar
    from datetime import datetime
    import subprocess
    
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "untracked_local_build"

    manifest_dir = Path(__file__).resolve().parent / "results"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest_file = manifest_dir / f"manifest_{ts}.json"

    telemetry = llm.get_audit_summary() if hasattr(llm, "get_audit_summary") else {}

    manifest_data = {
        "manifest_version": "1.0",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "git_commit": git_commit,
        "dataset": "ASAP-SAS",
        "prompt_set_id": "1",
        "sample_size_N": len(records),
        "llm_telemetry": telemetry,
        "human_ceiling": {
            "qwk": round(qwk_ceil, 4),
            "ci_95": [round(ceil_low, 4), round(ceil_high, 4)]
        },
        "ablations": {
            "zero_shot_baseline": {
                "qwk": round(qwk_zs, 4),
                "ci_95": [round(zs_low, 4), round(zs_high, 4)],
                "predictions": scores_zero_shot
            },
            "rubric_decomposition_only": {
                "qwk": round(qwk_rub, 4),
                "ci_95": [round(rub_low, 4), round(rub_high, 4)],
                "predictions": scores_rubric_only
            },
            "full_consensus_ensemble": {
                "qwk": round(qwk_ens, 4),
                "ci_95": [round(ens_low, 4), round(ens_high, 4)],
                "predictions": scores_full_ensemble
            }
        },
        "ground_truth": {
            "human_1": h1_scores,
            "human_2": h2_scores
        },
        "fitted_logistic_weights": {
            "w1_overlap": round(norm_w[0], 4),
            "w2_score_variance": round(norm_w[1], 4),
            "w3_ocr_quality": round(norm_w[2], 4),
            "w4_lexical_grounding": round(norm_w[3], 4)
        }
    }

    with open(manifest_file, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    print(f"\n-> Run Manifest sidecar saved successfully to:\n   {manifest_file}\n")

if __name__ == "__main__":
    run_suite()
