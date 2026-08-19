import json
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.db.models import StudentSubmission, GradingRecord, RubricVersion
from src.rag.vector_store import GradingVectorStore
from src.rag.query_router import QueryRouter
from src.core.llm_client import LLMClient
from src import config

class HybridRAGEngine:
    """
    Stage 8 & 9: Hybrid RAG & Conversational Query Engine
    Integrates ChromaDB vector retrieval for qualitative explanations and
    SQL aggregate calculations for analytical queries.
    """
    def __init__(self, db: Session, vector_store: GradingVectorStore, llm_client: Optional[LLMClient] = None):
        self.db = db
        self.vector_store = vector_store
        self.router = QueryRouter(llm_client)
        self.llm = llm_client or LLMClient()

    def query(self, user_query: str) -> Dict[str, Any]:
        route = self.router.classify_query(user_query)

        if route == "analytical":
            return self._handle_analytical_query(user_query)
        else:
            return self._handle_vector_rag_query(user_query, route)

    def _handle_analytical_query(self, query: str) -> Dict[str, Any]:
        """Calculates exact cohort metrics from SQLite."""
        total_submissions = self.db.query(StudentSubmission).count()
        avg_score_res = self.db.query(func.avg(GradingRecord.final_score)).scalar() or 0.0
        
        # Route breakdown counts
        auto_accept_count = self.db.query(GradingRecord).filter(GradingRecord.routing_decision == "auto_accept").count()
        spot_check_count = self.db.query(GradingRecord).filter(GradingRecord.routing_decision == "flag_for_spot_check").count()
        review_count = self.db.query(GradingRecord).filter(GradingRecord.routing_decision == "requires_review").count()
        override_count = self.db.query(GradingRecord).filter(GradingRecord.is_overridden == True).count()

        total_records = max(1, auto_accept_count + spot_check_count + review_count)
        auto_accept_pct = (auto_accept_count / total_records) * 100

        stats_summary = (
            f"**Analytical Cohort Statistics (SQL-Computed):**\n"
            f"- **Total Submissions Tracked:** {total_submissions}\n"
            f"- **Average Criterion Score:** {avg_score_res:.2f} pts\n"
            f"- **Auto-Accepted Decisions:** {auto_accept_count} ({auto_accept_pct:.1f}%)\n"
            f"- **Flagged for Spot-Check:** {spot_check_count}\n"
            f"- **Routed for Human Review:** {review_count}\n"
            f"- **Human Overrides Applied:** {override_count}"
        )

        return {
            "route": "analytical",
            "response": stats_summary,
            "sources": [{"type": "SQL Database", "table": "grading_records"}]
        }

    def _handle_vector_rag_query(self, query: str, route: str) -> Dict[str, Any]:
        """Performs semantic search over (answer x criterion) evidence chunks."""
        chunks = self.vector_store.query(query_text=query, n_results=3)

        if not chunks:
            # Fallback if vector index is empty yet
            return {
                "route": route,
                "response": "No specific grading records matched your search query in the index.",
                "sources": []
            }

        retrieved_context = "\n---\n".join([
            f"Record ID: {c['id']}\nContent: {c['document']}\nMetadata: {json.dumps(c['metadata'])}"
            for c in chunks
        ])

        system_prompt = (
            "Answer the instructor's or student's question using ONLY the retrieved grading records below. "
            "Cite which criterion and evidence span your answer is based on. If the retrieved records don't "
            "contain enough information to answer confidently, say so rather than guessing."
        )

        user_prompt = (
            f"Retrieved records:\n{retrieved_context}\n\n"
            f"Question: {query}"
        )

        answer = self.llm.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=config.SCORING_MODEL
        )

        return {
            "route": route,
            "response": answer,
            "sources": [c["metadata"] for c in chunks]
        }
