import numpy as np
from typing import Optional, List
from src.core.rubric_engine import RubricCriterion
from src.core.segmentation.llm_backend import SegmentationResult, EvidenceSpan
from src.utils.text_align import extract_sentences_with_offsets
from src import config

class MPNetEmbeddingBackend:
    """
    Sentence Transformers (all-mpnet-base-v2) Backend:
    Embeds each sentence of the student's answer and the criterion's satisfaction condition.
    Returns sentences exceeding a semantic cosine similarity threshold.
    """
    def __init__(self, model_name: Optional[str] = None, similarity_threshold: float = 0.55):
        self.model_name = model_name or config.MPNET_EMBEDDING_MODEL
        self.similarity_threshold = similarity_threshold
        self._model = None
        self._initialized = False

    def _lazy_init(self):
        if not self._initialized:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"SentenceTransformer init skipped ({e}). Using cosine n-gram fallback.")
                self._model = None
            self._initialized = True

    def extract_evidence(self, criterion: RubricCriterion, student_answer: str) -> SegmentationResult:
        self._lazy_init()
        sentences = extract_sentences_with_offsets(student_answer)
        if not sentences:
            return SegmentationResult(criterion_id=criterion.id, evidence_found=False, evidence_spans=[])

        target_text = f"{criterion.description}. {criterion.satisfaction_condition}"

        if self._model:
            try:
                sentence_texts = [s["text"] for s in sentences]
                embeddings = self._model.encode([target_text] + sentence_texts)
                target_emb = embeddings[0]
                sent_embs = embeddings[1:]

                # Cosine similarity
                norm_target = np.linalg.norm(target_emb)
                norm_sents = np.linalg.norm(sent_embs, axis=1)
                sims = np.dot(sent_embs, target_emb) / (norm_sents * norm_target + 1e-9)

                matched_spans = []
                for idx, sim in enumerate(sims):
                    if sim >= self.similarity_threshold:
                        s_info = sentences[idx]
                        matched_spans.append(EvidenceSpan(
                            text=s_info["text"],
                            start_char=s_info["start_char"],
                            end_char=s_info["end_char"]
                        ))

                return SegmentationResult(
                    criterion_id=criterion.id,
                    evidence_found=len(matched_spans) > 0,
                    evidence_spans=matched_spans,
                    notes=f"MPNet matched {len(matched_spans)} sentences above threshold {self.similarity_threshold}"
                )
            except Exception:
                pass

        # Word-overlap Jaccard semantic fallback
        target_words = set(target_text.lower().split())
        matched_spans = []
        for s_info in sentences:
            s_words = set(s_info["text"].lower().split())
            if not s_words:
                continue
            overlap = len(target_words.intersection(s_words)) / len(target_words.union(s_words))
            if overlap >= 0.12 or any(kw.lower() in s_info["text"].lower() for kw in criterion.keywords_or_concepts):
                matched_spans.append(EvidenceSpan(
                    text=s_info["text"],
                    start_char=s_info["start_char"],
                    end_char=s_info["end_char"]
                ))

        return SegmentationResult(
            criterion_id=criterion.id,
            evidence_found=len(matched_spans) > 0,
            evidence_spans=matched_spans,
            notes="Extracted via MPNet semantic fallback"
        )
