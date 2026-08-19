import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR
from src.core.rubric_engine import RubricCriterion
from src.core.segmentation.llm_backend import LLMSegmentationBackend
from src.core.segmentation.deberta_backend import DebertaQABackend
from src.core.segmentation.mpnet_backend import MPNetEmbeddingBackend
from src.utils.metrics import compute_character_jaccard

def run_segmentation_eval():
    squad_file = DATA_DIR / "squad" / "squad_v2_sample.json"
    if not squad_file.exists():
        print("SQuAD sample file missing.")
        return

    with open(squad_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"\n=======================================================")
    print(f"EVALUATION: Segmentation Backends on SQuAD 2.0")
    print(f"Testing {len(samples)} span queries...")
    print(f"=======================================================\n")

    llm = LLMSegmentationBackend()
    deberta = DebertaQABackend()
    mpnet = MPNetEmbeddingBackend()

    backends = {
        "LLM Backend": llm,
        "DeBERTa-v3 QA": deberta,
        "MPNet Embeddings": mpnet
    }

    for name, backend in backends.items():
        correct_extractions = 0
        correct_unanswerable = 0
        total_unanswerable = sum(1 for s in samples if s["is_impossible"])
        total_answerable = len(samples) - total_unanswerable

        for item in samples:
            crit = RubricCriterion(
                id=item["id"],
                description=item["question"],
                points=1.0,
                satisfaction_condition=item["question"],
                keywords_or_concepts=item["question"].replace("?", "").split()
            )
            res = backend.extract_evidence(crit, item["context"])

            if item["is_impossible"]:
                if not res.evidence_found or len(res.evidence_spans) == 0:
                    correct_unanswerable += 1
            else:
                if res.evidence_found and len(res.evidence_spans) > 0:
                    # Check if extracted span overlaps with ground truth
                    gold = item["answers"][0]
                    gold_span = (gold["answer_start"], gold["answer_start"] + len(gold["text"]))
                    pred_span = (res.evidence_spans[0].start_char, res.evidence_spans[0].end_char)
                    jacc = compute_character_jaccard(gold_span, pred_span)
                    if jacc > 0.3:
                        correct_extractions += 1

        ans_acc = (correct_extractions / total_answerable) * 100 if total_answerable else 0
        unans_acc = (correct_unanswerable / total_unanswerable) * 100 if total_unanswerable else 0

        print(f"Backend: {name}")
        print(f"  - Extractive Span Recall (Answerable): {ans_acc:.1f}% ({correct_extractions}/{total_answerable})")
        print(f"  - Zero-Evidence Precision (Unanswerable): {unans_acc:.1f}% ({correct_unanswerable}/{total_unanswerable})")
        print("-------------------------------------------------------")

if __name__ == "__main__":
    run_segmentation_eval()
