import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from src.core.llm_client import LLMClient
from src.core.rubric_engine import RubricEngine
from src.core.segmentation.ensemble import SegmentationEnsemble
from src.core.segmentation.llm_backend import LLMSegmentationBackend
from src.core.scorer import EvidenceGroundedScorer
from src.core.confidence_engine import ConfidenceEngine
from src.utils.metrics import compute_qwk

def evaluate_domain(name: str, file_path: Path, conf_engine: ConfidenceEngine, rubric_engine: RubricEngine):
    print(f"\n=================================================================")
    print(f" EVALUATING DOMAIN: {name.upper()}")
    print(f"=================================================================")
    
    if not file_path.exists():
        print(f"  [!] File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    q_text = data["question_text"]
    ref_ans = data["reference_answer"]
    total_marks = float(data["total_marks"])
    records = data.get("sample_records", [])

    print(f"Question: {q_text[:75]}...")
    print(f"Generating Domain Rubric...")
    rubric = rubric_engine.generate_rubric(
        question_id=data["question_id"],
        question_text=q_text,
        reference_answer=ref_ans,
        total_marks=total_marks
    )
    print(f"  -> Generated {len(rubric.criteria)} criteria (Max Points: {rubric.total_marks}):")
    for c in rubric.criteria:
        print(f"     * [{c.id}] ({c.points} pts): {c.description[:60]}...")

    h1_scores = []
    h2_scores = []
    ai_scores = []
    confidences = []

    print("\nGrading Student Responses:")
    for idx, rec in enumerate(records, 1):
        s_ans = rec["student_answer"]
        h1 = int(rec["human_score_1"])
        h2 = int(rec["human_score_2"])

        rep = conf_engine.grade_full_answer(
            student_id=int(rec['id']),
            rubric=rubric,
            student_answer=s_ans
        )

        h1_scores.append(h1)
        h2_scores.append(h2)
        ai_scores.append(round(rep.total_score))
        confidences.append(rep.composite_confidence)

        print(f"  [{idx:02d}/{len(records)}] Student #{rec['id']} | Human Avg: {(h1+h2)/2:.1f} | AI Score: {rep.total_score:.1f}/{rubric.total_marks} | Conf: {rep.composite_confidence:.2f} | Routing: {rep.overall_routing}")

    if len(h1_scores) > 1 and (max(h1_scores) > min(h1_scores) or max(ai_scores) > min(ai_scores)):
        qwk = compute_qwk(h1_scores, ai_scores)
        print(f"\n-> {name} Stand-Alone QWK vs Human Rater 1: {qwk:.4f}")
    else:
        print(f"\n-> {name} Prediction Mean: {sum(ai_scores)/len(ai_scores):.2f} (Variance too low for QWK computation)")

def main():
    print("=================================================================")
    print("      MULTI-DOMAIN CROSS-BENCHMARK EVALUATION HARNESS            ")
    print("=================================================================")

    llm = LLMClient()
    rubric_engine = RubricEngine(llm)
    ensemble = SegmentationEnsemble(LLMSegmentationBackend(llm))
    scorer = EvidenceGroundedScorer(llm)
    conf_engine = ConfidenceEngine(ensemble, scorer)

    base_dir = Path(__file__).resolve().parent.parent / "data"

    domains = [
        ("ASAP-SAS (Science)", base_dir / "asap_sas" / "prompt_set_1.json"),
        ("SciEntsBank (Physics / Electricity)", base_dir / "scientsbank.json"),
        ("CodeNet (Computer Science / Python)", base_dir / "codenet.json"),
        ("ASAP-AES (English Persuasive Essay)", base_dir / "asap_aes.json")
    ]

    for name, path in domains:
        evaluate_domain(name, path, conf_engine, rubric_engine)

if __name__ == "__main__":
    main()
