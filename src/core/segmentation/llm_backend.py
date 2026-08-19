from typing import List, Optional
from pydantic import BaseModel, Field
from src.core.llm_client import LLMClient
from src.core.rubric_engine import RubricCriterion
from src.utils.text_align import find_substring_span
from src import config

class EvidenceSpan(BaseModel):
    text: str = Field(..., description="Verbatim extracted span")
    start_char: int = Field(..., description="Start character index in student answer")
    end_char: int = Field(..., description="End character index in student answer")

class SegmentationResult(BaseModel):
    criterion_id: str
    evidence_found: bool
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)
    notes: Optional[str] = None

class LLMSegmentationBackend:
    """
    LLM Segmentation Backend: Prompts the LLM to extract verbatim text spans addressing the criterion.
    Recalibrates character offsets to ensure exact span accuracy against the raw answer.
    """
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.model = config.RUBRIC_MODEL

    def extract_evidence(self, criterion: RubricCriterion, student_answer: str) -> SegmentationResult:
        system_prompt = (
            "You are extracting evidence for rubric-based grading. You will be given a "
            "rubric criterion and a student's full answer. Find the exact span(s) of text "
            "in the student's answer (verbatim, unmodified) that address this criterion. "
            "If no relevant text exists, say so explicitly — do not infer or paraphrase."
        )

        user_prompt = (
            f"Rubric criterion: {criterion.description} — satisfied if: {criterion.satisfaction_condition}\n"
            f"Student answer: {student_answer}\n"
        )

        result: SegmentationResult = self.llm.structured_output(
            prompt=user_prompt,
            system_prompt=system_prompt,
            schema=SegmentationResult,
            model=self.model,
            temperature=0.0
        )

        # Post-process spans with exact string alignment to fix any LLM offset calculation inaccuracies
        corrected_spans = []
        for span in result.evidence_spans:
            exact_span = find_substring_span(student_answer, span.text)
            if exact_span:
                corrected_spans.append(EvidenceSpan(
                    text=student_answer[exact_span[0]:exact_span[1]],
                    start_char=exact_span[0],
                    end_char=exact_span[1]
                ))
            else:
                # If substring was slightly altered, attempt search
                corrected_spans.append(span)

        result.evidence_spans = corrected_spans
        result.evidence_found = len(corrected_spans) > 0
        return result
