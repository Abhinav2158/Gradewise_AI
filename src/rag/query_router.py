import re
from typing import Optional
from pydantic import BaseModel, Field
from src.core.llm_client import LLMClient
from src import config

class RouterClassification(BaseModel):
    query_type: str = Field(..., description="'explanatory', 'comparative', or 'analytical'")
    reasoning: Optional[str] = None

class QueryRouter:
    """
    Stage 8: RAG Query Router
    Directs analytical queries to Structured SQL and qualitative queries to Vector Retrieval.
    """
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.model = config.SCORING_MODEL

    def classify_query(self, query: str) -> str:
        # Fast regex heuristic rules for common analytical patterns
        analytical_keywords = ["average", "mean", "total", "percentage", "distribution", "how many", "count", "stats", "statistics", "highest score", "lowest score", "rate"]
        q_lower = query.lower()
        if any(kw in q_lower for kw in analytical_keywords):
            return "analytical"

        comparative_keywords = ["compare", "similar", "difference between", "versus", "vs", "alike"]
        if any(kw in q_lower for kw in comparative_keywords):
            return "comparative"

        system_prompt = (
            "Classify the incoming query as one of:\n"
            "- 'explanatory': asks why a specific answer/student got a specific score\n"
            "- 'comparative': asks to compare answers or find similar-scored answers\n"
            "- 'analytical': asks for aggregate stats (averages, disagreement rates, score distributions)\n\n"
            "Analytical queries should be routed to the structured SQL layer, not vector retrieval.\n"
            "Output strictly valid JSON with field 'query_type'."
        )

        user_prompt = f"Query: {query}"

        try:
            res: RouterClassification = self.llm.structured_output(
                prompt=user_prompt,
                system_prompt=system_prompt,
                schema=RouterClassification,
                model=self.model
            )
            if res.query_type in ["explanatory", "comparative", "analytical"]:
                return res.query_type
        except Exception:
            pass

        return "explanatory"
