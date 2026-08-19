import json
from typing import List, Dict, Any, Optional
from src import config

class GradingVectorStore:
    """
    Stage 8: RAG Indexing Layer
    Unit of indexing: (student_answer × criterion) pair containing:
    - Question ID and Student ID
    - Rubric criterion & satisfaction condition
    - Evidence spans extracted
    - Score & justification
    - Rubric version
    """
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_dir = persist_directory or config.CHROMA_PERSIST_DIR
        self._collection = None
        self._client = None
        self._fallback_records = []
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="grading_records",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"ChromaDB persistent client fallback mode: {e}")
            self._collection = None

    def index_record(self, doc_id: str, document_text: str, metadata: Dict[str, Any]):
        """Indexes a single (student_answer x criterion) record."""
        # Sanitize metadata for ChromaDB (no nested lists/dicts)
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = json.dumps(v)

        if self._collection:
            try:
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[document_text],
                    metadatas=[clean_meta]
                )
                return
            except Exception as e:
                pass

        # In-memory fallback
        self._fallback_records.append({
            "id": doc_id,
            "document": document_text,
            "metadata": clean_meta
        })

    def query(self, query_text: str, n_results: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Queries the vector index for matching grading chunks."""
        if self._collection:
            try:
                results = self._collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=filter_dict
                )
                formatted = []
                if results and results.get("documents") and results["documents"][0]:
                    for i in range(len(results["documents"][0])):
                        formatted.append({
                            "id": results["ids"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i]
                        })
                return formatted
            except Exception:
                pass

        # Simple keyword retrieval fallback
        query_words = set(query_text.lower().split())
        scored = []
        for rec in self._fallback_records:
            doc_words = set(rec["document"].lower().split())
            score = len(query_words.intersection(doc_words))
            scored.append((score, rec))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:n_results]]
