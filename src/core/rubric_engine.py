import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.core.llm_client import LLMClient
from src import config

class RubricCriterion(BaseModel):
    id: str = Field(..., description="Unique criterion ID, e.g. crit_1")
    description: str = Field(..., description="Atomic criterion description")
    points: float = Field(..., description="Point value for this criterion")
    satisfaction_condition: str = Field(..., description="Minimal condition/fact/calculation required to earn points")
    keywords_or_concepts: List[str] = Field(default_factory=list, description="Key concepts, numbers, or terms")

class RubricSchema(BaseModel):
    question_id: str
    total_marks: float
    criteria: List[RubricCriterion]

class RubricRefinementResult(BaseModel):
    proposed_change: str = Field(..., description="'add_criterion', 'modify_criterion', or 'reallocate_points'")
    details: Dict[str, Any] = Field(..., description="Specific details of the updated or new criterion")
    rationale: str = Field(..., description="Reasoning justifying the amendment")
    estimated_affected_answers: int = Field(default=0, description="Estimated number of past answers affected")

def is_math_or_numerical(text: str) -> bool:
    """Detects if a question is mathematical, statistical, or numerical."""
    math_keywords = [
        "probability", "mean", "standard deviation", "normal distribution", "calculate",
        "solve", "compute", "z-score", "variance", "integral", "derivative", "equation",
        "months", "percentage", "ratio", "median", "average", "p(", "p-value"
    ]
    t_lower = text.lower()
    return any(k in t_lower for k in math_keywords)

class RubricEngine:
    """
    Manages Stage 1 (Rubric Generation) and Stage 6 (Rubric Refinement).
    Auto-specializes for Math/Numerical, Coding, and Subjective questions.
    """
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.model = config.RUBRIC_MODEL

    def generate_reference_answer(self, question_text: str, total_marks: float = 5.0) -> str:
        """
        Auto-generates an authoritative reference answer or step-by-step numerical solution.
        """
        if is_math_or_numerical(question_text):
            system_prompt = (
                "You are an expert mathematics and statistics professor. Solve the given numerical problem step-by-step:\n"
                "1. State the given parameters (e.g. mean, standard deviation, target bounds).\n"
                "2. Show the mathematical formula (e.g. Z = (X - μ) / σ).\n"
                "3. Calculate intermediate values.\n"
                "4. Compute the final numerical probability/value clearly."
            )
        else:
            system_prompt = (
                "You are a master educator and exam subject matter expert. Given an exam question and point value, "
                "write an authoritative, textbook-accurate, concise model reference answer that covers all essential concepts."
            )
        
        user_prompt = f"Question: {question_text}\nTotal Marks: {total_marks}\nProvide a complete reference solution:"
        ans = self.llm.complete(prompt=user_prompt, system_prompt=system_prompt, model=self.model, temperature=0.1)
        return ans.strip() if ans else f"Reference solution for: {question_text}"

    def generate_rubric(self, question_id: str, question_text: str, reference_answer: str, total_marks: float) -> RubricSchema:
        """
        Stage 1: Generates atomic, independently-gradable rubric criteria.
        Specializes for Math / Numerical / Coding / Subjective problems.
        """
        if is_math_or_numerical(question_text):
            system_prompt = (
                "You are an expert mathematics and statistics exam evaluator. For the numerical/probability problem, "
                "produce an atomic grading rubric assessing:\n"
                "1. Formula Setup & Parameter Identification: Stating correct parameters (e.g. mean μ, std dev σ) and formulas (e.g. Z = (X-μ)/σ).\n"
                "2. Intermediate Calculations: Correct computation of intermediate values (e.g. Z-scores, normal CDF conversions).\n"
                "3. Final Numerical Result & Interpretation: Correct final probability / numerical answer with unit/percentage.\n"
                "Each criterion must have explicit points summing to total marks and clear numerical conditions."
            )
        elif any(k in question_text.lower() for k in ["code", "program", "function", "algorithm", "python", "java", "c++", "def "]):
            system_prompt = (
                "You are an expert computer science evaluator. Produce an atomic rubric assessing:\n"
                "1. Algorithmic Logic & Correct Output.\n"
                "2. Syntax, Function Signature & Edge Cases.\n"
                "3. Efficiency & Time Complexity (Big-O).\n"
                "Each criterion must have explicit points summing to total marks."
            )
        else:
            system_prompt = (
                "You are an exam rubric designer. Produce a rubric decomposed into atomic, independently-gradable criteria, "
                "each with explicit points summing to total marks and precise satisfaction conditions."
            )

        user_prompt = (
            f"Question ID: {question_id}\n"
            f"Question: {question_text}\n"
            f"Reference/model answer: {reference_answer}\n"
            f"Total marks: {total_marks}\n"
        )

        return self.llm.structured_output(
            prompt=user_prompt,
            system_prompt=system_prompt,
            schema=RubricSchema,
            model=self.model,
            temperature=0.1
        )

    def propose_refinement(self, current_rubric: RubricSchema, instructor_flag_text: str, student_answer_excerpt: str) -> RubricRefinementResult:
        """
        Stage 6: Proposes minimal rubric amendments when an instructor flags a novel valid point.
        """
        system_prompt = (
            "An instructor has flagged that a student made a valid point not currently covered by any rubric criterion. "
            "Propose a MINIMAL rubric amendment."
        )
        user_prompt = (
            f"Current rubric: {current_rubric.model_dump_json(indent=2)}\n"
            f"Instructor note: {instructor_flag_text}\n"
            f"Student excerpt: {student_answer_excerpt}\n"
        )
        return self.llm.structured_output(
            prompt=user_prompt,
            system_prompt=system_prompt,
            schema=RubricRefinementResult,
            model=self.model,
            temperature=0.1
        )
