import numpy as np
from sklearn.metrics import cohen_kappa_score
from typing import List, Tuple, Dict, Any

def compute_qwk(y_true: List[int], y_pred: List[int]) -> float:
    """
    Computes Quadratic Weighted Kappa (QWK) between true and predicted integer scores.
    This is the standard metric used in ASAP-SAS automated grading benchmarks.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

def compute_character_jaccard(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
    """
    Computes Jaccard index between two character intervals (start_char, end_char).
    """
    start_a, end_a = span_a
    start_b, end_b = span_b

    # If either interval is empty or invalid
    if start_a >= end_a or start_b >= end_b:
        return 0.0

    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0, inter_end - inter_start)

    union = (end_a - start_a) + (end_b - start_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def compute_span_overlap_agreement(spans_list: List[List[Tuple[int, int]]]) -> float:
    """
    Computes pairwise average Jaccard agreement across multiple backends' extracted spans.
    spans_list: list of spans per backend, e.g. [[(0, 25)], [(0, 28)], [(5, 30)]]
    """
    valid_backends = [s for s in spans_list if len(s) > 0]
    
    # If no backend found evidence: agreement is 1.0 (all agreed no evidence)
    if len(valid_backends) == 0 and all(len(s) == 0 for s in spans_list):
        return 1.0
    
    # If some found evidence and some did not
    if len(valid_backends) < len(spans_list) and len(valid_backends) > 0:
        base_penalty = len(valid_backends) / len(spans_list)
    else:
        base_penalty = 1.0

    if len(valid_backends) < 2:
        return base_penalty * 0.5

    pairwise_scores = []
    for i in range(len(valid_backends)):
        for j in range(i + 1, len(valid_backends)):
            # Max overlap across spans of backend i and backend j
            max_jaccard = 0.0
            for span1 in valid_backends[i]:
                for span2 in valid_backends[j]:
                    jacc = compute_character_jaccard(span1, span2)
                    if jacc > max_jaccard:
                        max_jaccard = jacc
            pairwise_scores.append(max_jaccard)

    return float(np.mean(pairwise_scores)) * base_penalty if pairwise_scores else 0.0
