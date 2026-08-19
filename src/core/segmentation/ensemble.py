from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from src.core.rubric_engine import RubricCriterion
from src.core.segmentation.llm_backend import LLMSegmentationBackend, SegmentationResult, EvidenceSpan
from src.core.segmentation.deberta_backend import DebertaQABackend
from src.core.segmentation.mpnet_backend import MPNetEmbeddingBackend
from src.utils.metrics import compute_span_overlap_agreement

class EnsembleSegmentationOutput(BaseModel):
    criterion_id: str
    llm_result: SegmentationResult
    deberta_result: SegmentationResult
    mpnet_result: SegmentationResult
    combined_evidence_spans: List[EvidenceSpan]
    evidence_found_unanimous: bool
    evidence_found_majority: bool
    span_overlap_agreement: float = Field(..., description="Pairwise Jaccard overlap between backends [0.0 - 1.0]")

class SegmentationEnsemble:
    """
    Stage 2: Answer Segmentation Ensemble
    Runs LLM, DeBERTa, and MPNet backends in parallel and evaluates cross-model span overlap.
    """
    def __init__(self, llm_backend: Optional[LLMSegmentationBackend] = None):
        self.llm_backend = llm_backend or LLMSegmentationBackend()
        self.deberta_backend = DebertaQABackend()
        self.mpnet_backend = MPNetEmbeddingBackend()

    def segment_answer(self, criterion: RubricCriterion, student_answer: str) -> EnsembleSegmentationOutput:
        # Run all 3 backends
        llm_res = self.llm_backend.extract_evidence(criterion, student_answer)
        deb_res = self.deberta_backend.extract_evidence(criterion, student_answer)
        mpn_res = self.mpnet_backend.extract_evidence(criterion, student_answer)

        # Spans per backend
        llm_spans = [(s.start_char, s.end_char) for s in llm_res.evidence_spans]
        deb_spans = [(s.start_char, s.end_char) for s in deb_res.evidence_spans]
        mpn_spans = [(s.start_char, s.end_char) for s in mpn_res.evidence_spans]

        # Agreement metrics
        overlap_score = compute_span_overlap_agreement([llm_spans, deb_spans, mpn_spans])
        
        flags = [llm_res.evidence_found, deb_res.evidence_found, mpn_res.evidence_found]
        unanimous = all(flags) or not any(flags)
        majority = sum(flags) >= 2

        # Combine distinct evidence spans without duplicates
        combined = []
        seen = set()
        for backend_res in [llm_res, deb_res, mpn_res]:
            for span in backend_res.evidence_spans:
                span_key = (span.start_char, span.end_char)
                if span_key not in seen:
                    seen.add(span_key)
                    combined.append(span)

        return EnsembleSegmentationOutput(
            criterion_id=criterion.id,
            llm_result=llm_res,
            deberta_result=deb_res,
            mpnet_result=mpn_res,
            combined_evidence_spans=combined,
            evidence_found_unanimous=unanimous,
            evidence_found_majority=majority,
            span_overlap_agreement=overlap_score
        )
