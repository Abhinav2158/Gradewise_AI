from typing import Optional, List
from src.core.rubric_engine import RubricCriterion
from src.core.segmentation.llm_backend import SegmentationResult, EvidenceSpan
from src.utils.text_align import find_substring_span
from src import config

class DebertaQABackend:
    """
    DeBERTa-v3 Extractive QA Backend:
    Treats `criterion.satisfaction_condition` as the question and `student_answer` as the context.
    Uses Hugging Face AutoModelForQuestionAnswering, with a fast fallback if running in lightweight environments.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.DEBERTA_QA_MODEL
        self._tokenizer = None
        self._model = None
        self._initialized = False

    def _lazy_init(self):
        if not self._initialized:
            try:
                from transformers import AutoTokenizer, AutoModelForQuestionAnswering
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            except Exception as e:
                print(f"DeBERTa pipeline init failed or model not cached ({e}). Using fast heuristic extractive QA.")
                self._tokenizer = None
                self._model = None
            self._initialized = True

    def extract_evidence(self, criterion: RubricCriterion, student_answer: str) -> SegmentationResult:
        self._lazy_init()
        question = f"What states that: {criterion.satisfaction_condition}?"

        if self._model and self._tokenizer:
            try:
                import torch
                inputs = self._tokenizer(question, student_answer, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self._model(**inputs)
                
                start_idx = torch.argmax(outputs.start_logits, dim=1).item()
                end_idx = torch.argmax(outputs.end_logits, dim=1).item()

                if end_idx >= start_idx:
                    input_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
                    ans_text = self._tokenizer.decode(input_ids, skip_special_tokens=True).strip()
                    if ans_text and len(ans_text) > 3 and ans_text.lower() not in question.lower():
                        span_coords = find_substring_span(student_answer, ans_text)
                        if span_coords:
                            return SegmentationResult(
                                criterion_id=criterion.id,
                                evidence_found=True,
                                evidence_spans=[EvidenceSpan(text=student_answer[span_coords[0]:span_coords[1]], start_char=span_coords[0], end_char=span_coords[1])],
                                notes="Extracted via DeBERTa-v3 QA model"
                            )
            except Exception as e:
                pass

        # Robust extractive heuristic fallback
        return self._heuristic_span_extractor(criterion, student_answer)

    def _heuristic_span_extractor(self, criterion: RubricCriterion, student_answer: str) -> SegmentationResult:
        keywords = criterion.keywords_or_concepts or criterion.satisfaction_condition.lower().split()
        matched_spans = []

        import re
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', student_answer) if s.strip()]
        for sentence in sentences:
            sentence_lower = sentence.lower()
            hit_count = sum(1 for kw in keywords if kw.lower() in sentence_lower)
            if hit_count >= 1:
                span_coords = find_substring_span(student_answer, sentence)
                if span_coords:
                    matched_spans.append(EvidenceSpan(
                        text=sentence,
                        start_char=span_coords[0],
                        end_char=span_coords[1]
                    ))

        return SegmentationResult(
            criterion_id=criterion.id,
            evidence_found=len(matched_spans) > 0,
            evidence_spans=matched_spans[:1], # Top extractive match
            notes="Extracted via DeBERTa QA heuristic fallback"
        )
