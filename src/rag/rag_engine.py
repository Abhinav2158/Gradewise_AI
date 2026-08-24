import json
import re
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from src.db.models import StudentSubmission, GradingRecord
from src.rag.query_router import QueryRouter
from src.rag.vector_store import GradingVectorStore
from src.core.llm_client import LLMClient
from src import config

class HybridRAGEngine:
    """
    Stage 7 & Tab 5: Hybrid RAG & SQL Analytics Engine
    Intelligently routes analytical cohort questions to SQL queries (aggregated per student and per question)
    and qualitative reasoning / justification queries to grounded semantic search with exact entity validation.
    """
    def __init__(self, db: Session, vector_store: GradingVectorStore, llm_client: Optional[LLMClient] = None):
        self.db = db
        self.vector_store = vector_store
        self.router = QueryRouter(llm_client)
        self.llm = llm_client or LLMClient()

    def query(self, user_query: str) -> Dict[str, Any]:
        route = self.router.classify_query(user_query)

        # Check if query asks for a specific student
        st_match = re.search(r"student\s*#?\s*(\w+)", user_query, re.IGNORECASE)
        if st_match:
            target_st = st_match.group(1)
            # Check if this student exists in DB
            subs = self.db.query(StudentSubmission).filter(
                StudentSubmission.student_id.cast(str) == str(target_st)
            ).all()

            if not subs:
                available_ids = [str(r[0]) for r in self.db.query(distinct(StudentSubmission.student_id)).all()]
                if available_ids:
                    avail_str = ", ".join([f"**Student #{sid}**" for sid in available_ids])
                    return {
                        "route": "explanatory",
                        "response": f"❌ **Student #{target_st} was not found in the database.**\n\nCurrently graded students in your database: {avail_str}. Please select or ask about one of these students.",
                        "sources": [{"type": "SQL Database", "table": "student_submissions", "available_students": available_ids}]
                    }
                else:
                    return {
                        "route": "explanatory",
                        "response": f"❌ **Student #{target_st} was not found.** The database is currently empty. Please grade a student answer sheet in **'📄 Full Exam & Batch PDF Grading'** or **'⚡ Live Grading Console'** first.",
                        "sources": []
                    }

            # Student exists in DB! Gather their exact submissions and grading records
            sub_ids = [s.id for s in subs]
            records = self.db.query(GradingRecord).filter(GradingRecord.submission_id.in_(sub_ids)).all()

            total_earned = sum(r.final_score for r in records)
            total_max = sum(r.max_points for r in records)
            pct = (total_earned / max(0.1, total_max)) * 100

            resp_md = (
                f"### 📋 Grading Breakdown for Student #{target_st}\n\n"
                f"- **Overall Exam Score:** `{total_earned:.1f} / {total_max:.1f} pts` (**{pct:.1f}%**)\n"
                f"- **Total Questions Answered:** `{len(subs)}`\n\n"
                f"#### 🔍 Question-by-Question Diagnostic:\n"
            )

            for s in subs:
                s_recs = [r for r in records if r.submission_id == s.id]
                q_score = sum(r.final_score for r in s_recs)
                q_max = sum(r.max_points for r in s_recs)
                resp_md += f"**Question `{s.question_id}` (Score: {q_score:.1f}/{q_max:.1f} pts):**\n"
                resp_md += f"> *Submitted Answer:* \"{s.answer_text}\"\n\n"
                for r in s_recs:
                    resp_md += f"- **Checkpoint `{r.criterion_id}` ({r.final_score}/{r.max_points} pts):** {r.justification}\n"
                resp_md += "\n"

            return {
                "route": "explanatory",
                "response": resp_md,
                "sources": [{"type": "SQL Database", "student_id": target_st, "records_count": len(records)}]
            }

        if route == "analytical":
            return self._handle_analytical_query(user_query)
        else:
            return self._handle_vector_rag_query(user_query, route)

    def _handle_analytical_query(self, query: str) -> Dict[str, Any]:
        """Calculates exact cohort metrics aggregated per Student and per Question from SQLite."""
        distinct_students = self.db.query(func.count(distinct(StudentSubmission.student_id))).scalar() or 0
        distinct_questions = self.db.query(func.count(distinct(StudentSubmission.question_id))).scalar() or 0
        total_question_answers = self.db.query(StudentSubmission).count()
        total_criterion_records = self.db.query(GradingRecord).count()

        student_scores = (
            self.db.query(
                StudentSubmission.student_id,
                func.sum(GradingRecord.final_score).label("total_score"),
                func.sum(GradingRecord.max_points).label("max_score")
            )
            .join(GradingRecord, GradingRecord.submission_id == StudentSubmission.id)
            .group_by(StudentSubmission.student_id)
            .all()
        )

        avg_student_score = 0.0
        avg_student_max = 0.0
        avg_student_pct = 0.0

        if student_scores:
            avg_student_score = sum(s.total_score for s in student_scores) / len(student_scores)
            avg_student_max = sum(s.max_score for s in student_scores) / len(student_scores)
            avg_student_pct = (avg_student_score / max(1.0, avg_student_max)) * 100

        auto_accept_count = self.db.query(GradingRecord).filter(GradingRecord.routing_decision == "auto_accept").count()
        spot_check_count = self.db.query(GradingRecord).filter(GradingRecord.routing_decision == "flag_for_spot_check").count()
        review_count = self.db.query(GradingRecord).filter(GradingRecord.routing_decision == "requires_review").count()
        override_count = self.db.query(GradingRecord).filter(GradingRecord.is_overridden == True).count()

        total_decisions = max(1, auto_accept_count + spot_check_count + review_count)
        auto_accept_pct = (auto_accept_count / total_decisions) * 100

        student_rows_md = ""
        for s in student_scores:
            pct = (s.total_score / max(0.1, s.max_score)) * 100
            student_rows_md += f"| **Student #{s.student_id}** | `{s.total_score:.1f} / {s.max_score:.1f} pts` | `{pct:.1f}%` |\n"

        stats_summary = (
            f"### 📊 Student Cohort Analytics (SQL-Computed)\n\n"
            f"#### 👤 High-Level Student Summary:\n"
            f"- **Unique Students Graded:** `{distinct_students}`\n"
            f"- **Exam Questions Evaluated:** `{distinct_questions}` distinct items\n"
            f"- **Average Student Exam Score:** `{avg_student_score:.1f} / {avg_student_max:.1f} pts` (**{avg_student_pct:.1f}%**)\n"
            f"- **Total Question Answers Evaluated:** `{total_question_answers}`\n"
            f"- **Atomic Rubric Checkpoint Records:** `{total_criterion_records}`\n\n"
        )

        if student_scores:
            stats_summary += (
                f"#### 📋 Itemized Breakdown Per Student ID:\n"
                f"| Student ID | Total Exam Score | Percentage |\n"
                f"| :--- | :---: | :---: |\n"
                f"{student_rows_md}\n"
            )

        stats_summary += (
            f"#### 🚦 Quality & Confidence Routing:\n"
            f"- **Auto-Accepted:** `{auto_accept_count}` ({auto_accept_pct:.1f}%)\n"
            f"- **Flagged for Teacher Spot-Check:** `{spot_check_count}`\n"
            f"- **Requires Human Review:** `{review_count}`\n"
            f"- **Instructor Overrides Applied:** `{override_count}`"
        )

        return {
            "route": "analytical",
            "response": stats_summary,
            "sources": [{"type": "SQL Database", "table": "student_submissions & grading_records"}]
        }

    def _handle_vector_rag_query(self, query: str, route: str) -> Dict[str, Any]:
        """Performs semantic search over (answer x criterion) evidence chunks."""
        chunks = self.vector_store.query(query_text=query, n_results=3)

        if not chunks:
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
