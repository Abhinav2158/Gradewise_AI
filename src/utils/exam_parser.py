import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ExamQuestion(BaseModel):
    number: int
    id: str
    title: str
    text: str
    max_marks: float = 5.0
    reference_answer: Optional[str] = None
    question_type: str = "subjective"  # "math", "code", "subjective"

class ExamQuestionPaper(BaseModel):
    title: str = "Exam Paper"
    total_exam_marks: float = 0.0
    questions: List[ExamQuestion] = Field(default_factory=list)

def detect_question_type(text: str) -> str:
    """Categorizes a question as 'math', 'code', or 'subjective'."""
    t_lower = text.lower()
    
    # 1. Math / Numerical / Statistical Check
    math_signals = [
        "probability", "mean", "standard deviation", "normal distribution", "z-score",
        "calculate the probability", "variance", "integral", "derivative", "solve for x",
        "expected value", "p(x", "p(z", "binomial", "poisson", "median", "standard error"
    ]
    if any(sig in t_lower for sig in math_signals):
        return "math"

    # 2. Programming / Coding Check (Requires specific coding intent)
    code_signals = [
        "write a python program", "write a python function", "write a c++ program",
        "write a java function", "write a function to", "write a script to", "write a sql query",
        "time complexity of", "space complexity of", "implement an algorithm", "implement a function",
        "def ", "class ", "#include", "public static void"
    ]
    if any(sig in t_lower for sig in code_signals):
        return "code"

    return "subjective"

def detect_student_answer_type(text: str) -> str:
    """Detects whether student answer content is Code, Math, or Subjective Text."""
    # Look for code syntax signatures
    code_regex = [
        r'def\s+\w+\s*\(.*?\)\s*:',
        r'class\s+\w+\s*[\(\{:]',
        r'#include\s*<.*?>',
        r'public\s+(?:static\s+)?(?:void|int|String|boolean)\s+\w+\s*\(',
        r'System\.out\.println\s*\(',
        r'console\.log\s*\(',
        r'SELECT\s+.*?\s+FROM\s+',
        r'for\s*\(\s*int\s+\w+\s*='
    ]
    for pattern in code_regex:
        if re.search(pattern, text):
            return "code"

    # Look for math working
    if any(k in text.lower() for k in ["z =", "z1 =", "z2 =", "p(", "μ =", "σ =", "mean =", "p-value"]):
        return "math"

    return "subjective"

def extract_questions_from_text(paper_text: str) -> ExamQuestionPaper:
    """
    Parses a multi-question exam paper text, identifying questions,
    categorizing their types, and extracting marks.
    """
    if not paper_text.strip():
        return ExamQuestionPaper()

    pattern = r'(?:^|\n)\s*(?:Q(?:uestion)?\s*(\d+)[\.:\)\-]|Problem\s*(\d+)[\.:\)\-]|Task\s*(\d+)[\.:\)\-]|(\d+)[\.\)\-]\s+)(.*?)(?=(?:\n\s*(?:Q(?:uestion)?\s*\d+[\.:\)\-]|Problem\s*\d+[\.:\)\-]|Task\s*\d+[\.:\)\-]|\d+[\.\)\-]\s+))|$)'
    matches = list(re.finditer(pattern, paper_text, re.DOTALL | re.IGNORECASE))

    questions = []
    total_marks = 0.0

    if matches:
        for idx, match in enumerate(matches, 1):
            q_num_str = match.group(1) or match.group(2) or match.group(3) or match.group(4) or str(idx)
            q_num = int(q_num_str) if q_num_str.isdigit() else idx
            raw_q_text = match.group(5).strip()

            if not raw_q_text:
                continue

            mark_match = re.search(r'[\[\(](\d+(?:\.\d+)?)\s*(?:marks?|pts?|points?)[\]\)]', raw_q_text, re.IGNORECASE)
            q_marks = float(mark_match.group(1)) if mark_match else 5.0

            clean_text = re.sub(r'[\[\(]\d+(?:\.\d+)?\s*(?:marks?|pts?|points?)[\]\)]', '', raw_q_text, flags=re.IGNORECASE).strip()

            q_type = detect_question_type(clean_text)

            questions.append(ExamQuestion(
                number=q_num,
                id=f"Q_{q_num}",
                title=f"Question {q_num}",
                text=clean_text,
                max_marks=q_marks,
                question_type=q_type
            ))
            total_marks += q_marks
    else:
        q_type = detect_question_type(paper_text.strip())
        questions.append(ExamQuestion(
            number=1,
            id="Q_1",
            title="Question 1",
            text=paper_text.strip(),
            max_marks=5.0,
            question_type=q_type
        ))
        total_marks = 5.0

    return ExamQuestionPaper(
        title="Parsed Exam Paper",
        total_exam_marks=total_marks,
        questions=questions
    )

def split_student_answers_for_exam(student_full_text: str, questions: List[ExamQuestion]) -> Dict[str, str]:
    """
    Intelligently segments student answers and maps each specific answer section to its question.
    """
    mapped_answers = {}
    if not student_full_text.strip() or not questions:
        return mapped_answers

    if len(questions) == 1:
        mapped_answers[questions[0].id] = student_full_text.strip()
        return mapped_answers

    # Strategy 1: Explicit Question/Answer Tagging
    pattern = r'(?:^|\n)\s*(?:Ans(?:wer)?\s*(\d+)[\.:\)\-]|Sol(?:ution)?\s*(\d+)[\.:\)\-]|Q(?:uestion)?\s*(\d+)[\.:\)\-]|(\d+)[\.\)\-]\s+)(.*?)(?=(?:\n\s*(?:Ans(?:wer)?\s*\d+[\.:\)\-]|Sol(?:ution)?\s*\d+[\.:\)\-]|Q(?:uestion)?\s*\d+[\.:\)\-]|\d+[\.\)\-]\s+))|$)'
    matches = list(re.finditer(pattern, student_full_text, re.DOTALL | re.IGNORECASE))

    if matches and len(matches) >= 2:
        for match in matches:
            q_num_str = match.group(1) or match.group(2) or match.group(3) or match.group(4)
            if q_num_str and q_num_str.isdigit():
                q_id = f"Q_{int(q_num_str)}"
                ans_content = match.group(5).strip()
                if ans_content:
                    mapped_answers[q_id] = ans_content

    # Strategy 2: If untagged, partition by distinct paragraphs/sections
    unmapped = [q for q in questions if q.id not in mapped_answers]
    if unmapped:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', student_full_text) if len(p.strip()) > 10]
        if len(paragraphs) >= len(questions):
            chunk_size = max(1, len(paragraphs) // len(questions))
            for idx, q in enumerate(questions):
                if q.id not in mapped_answers:
                    start_p = idx * chunk_size
                    end_p = (idx + 1) * chunk_size if idx < len(questions) - 1 else len(paragraphs)
                    assigned_chunk = "\n\n".join(paragraphs[start_p:end_p]).strip()
                    mapped_answers[q.id] = assigned_chunk or paragraphs[min(idx, len(paragraphs)-1)]
        else:
            for q in unmapped:
                mapped_answers[q.id] = student_full_text.strip()

    return mapped_answers
