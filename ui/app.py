import os
import sys
import json
import io
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
from src.config import DATA_DIR
from src.db.database import init_db, get_db
from src.db.models import RubricVersion, StudentSubmission, GradingRecord, AuditTrail
from src.core.llm_client import LLMClient
from src.core.rubric_engine import RubricEngine, RubricSchema, RubricCriterion
from src.core.segmentation.llm_backend import LLMSegmentationBackend
from src.core.segmentation.ensemble import SegmentationEnsemble
from src.core.scorer import EvidenceGroundedScorer
from src.core.confidence_engine import ConfidenceEngine
from src.core.rescoring_queue import RescoringQueue
from src.rag.vector_store import GradingVectorStore
from src.rag.rag_engine import HybridRAGEngine
from src.utils.pdf_parser import parse_uploaded_file, ParsedDocument
from src.utils.exam_parser import extract_questions_from_text, split_student_answers_for_exam, ExamQuestionPaper, ExamQuestion

# Initialize DB tables
init_db()

st.set_page_config(
    page_title="AI Exam Grading Suite | Consensus & HITL",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Theme & Visual Styling
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }

    /* Top Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.2);
        color: #FFFFFF;
        position: relative;
        overflow: hidden;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(129, 140, 248, 0.25) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #FFFFFF 0%, #E0E7FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #C7D2FE;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.5;
    }
    
    /* Modern Glassmorphic Step Cards */
    .glass-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.3rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.02) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -4px rgba(0, 0, 0, 0.03) !important;
    }
    
    /* File Uploader Light Theme Overrides */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #F8FAFC !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 1.2rem !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #6366F1 !important;
        background-color: #F1F5F9 !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #1E293B !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #EEF2FF !important;
        border-color: #818CF8 !important;
        color: #4338CA !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span, 
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #475569 !important;
    }
    [data-testid="stFileUploaderFile"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }
    [data-testid="stFileUploaderFile"] * {
        color: #0F172A !important;
    }
    
    .card-header-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #4F46E5;
        background: #EEF2FF;
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0 1.5rem 0;
    }
    .kpi-box {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .kpi-num {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E1B4B;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.2rem;
    }

    /* Custom Badges */
    .badge-auto {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.25);
        display: inline-block;
    }
    .badge-spot {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
        color: #FFFFFF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.25);
        display: inline-block;
    }
    .badge-review {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: #FFFFFF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: 0 2px 6px rgba(239, 68, 68, 0.25);
        display: inline-block;
    }
    
    /* Evidence Highlighting Span */
    .evidence-highlight {
        background: #FEF08A;
        border-bottom: 2px solid #EAB308;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        color: #713F12;
    }
    
    /* Sidebar High-Contrast Text Fix */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] [data-testid="stRadio"] span,
    [data-testid="stSidebar"] [data-testid="stRadio"] div {
        color: #FFFFFF !important;
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        color: #A5B4FC !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset
@st.cache_data
def load_asap_dataset(set_id: str):
    path = DATA_DIR / "asap_sas" / f"asap_set_{set_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.2rem 0 0.8rem 0;">
    <div style="background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); width: 56px; height: 56px; border-radius: 14px; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 28px; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);">🎓</div>
    <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; margin-top: 0.75rem; letter-spacing: -0.02em;">GRADEWISE AI</div>
    <div style="font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em;">Fault-Tolerant Grading Suite</div>
</div>
<hr style="border-color: #334155; margin: 0.5rem 0 1.2rem 0;">
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "NAVIGATION",
    [
        "📄 Full Exam & Batch PDF Grading",
        "⚡ Live Interactive Grading Console",
        "👨‍🏫 Instructor Review Queue (HITL)",
        "🔄 Versioned Rubrics & Re-scoring",
        "💬 Hybrid RAG & SQL Analytics"
    ]
)

st.sidebar.markdown("""
<div style="margin-top: 3rem; padding: 1rem; background: #1E293B; border-radius: 10px; border: 1px solid #334155;">
    <div style="font-size: 0.75rem; font-weight: 700; color: #A5B4FC; text-transform: uppercase;">Active Ensemble</div>
    <div style="font-size: 0.82rem; color: #E2E8F0; margin-top: 4px;">• LLM Verbatim Extractor</div>
    <div style="font-size: 0.82rem; color: #E2E8F0;">• DeBERTa-v3 QA Model</div>
    <div style="font-size: 0.82rem; color: #E2E8F0;">• MPNet Semantic Embeddings</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_services():
    db = get_db()
    llm = LLMClient()
    rubric_engine = RubricEngine(llm)
    ensemble = SegmentationEnsemble(LLMSegmentationBackend(llm))
    scorer = EvidenceGroundedScorer(llm)
    confidence_engine = ConfidenceEngine(ensemble, scorer)
    vector_store = GradingVectorStore()
    rag_engine = HybridRAGEngine(db, vector_store, llm)
    return db, llm, rubric_engine, ensemble, scorer, confidence_engine, vector_store, rag_engine

db, llm, rubric_engine, ensemble, scorer, confidence_engine, vector_store, rag_engine = get_services()

# -------------------------------------------------------------
# TAB 1: FULL EXAM & BATCH PDF GRADING
# -------------------------------------------------------------
if nav_choice == "📄 Full Exam & Batch PDF Grading":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📄 Multi-Question Exam Paper & Batch Grading</div>
        <div class="hero-subtitle">Upload full multi-question exam PDFs (Q1, Q2, Q3...) with automatic mark detection and grade student solutions with character-level evidence grounding.</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-tag">STEP 1 OF 3</div>
            <div class="card-title">Exam Question Paper</div>
        """, unsafe_allow_html=True)
        
        exam_input_mode = st.radio("Provide Question Paper via:", ["Upload Full Exam PDF", "Type/Paste Exam Questions"], horizontal=True)
        raw_paper_text = ""
        
        if exam_input_mode == "Upload Full Exam PDF":
            q_file = st.file_uploader("Upload Question Paper (PDF/TXT):", type=["pdf", "txt"], key="exam_pdf")
            if q_file:
                parsed_q = parse_uploaded_file(q_file, llm)
                if parsed_q.error:
                    st.error(f"⚠️ {parsed_q.error}")
                elif parsed_q.is_handwritten_or_scanned:
                    st.info(f" Scanned / Image PDF detected (`{q_file.name}`). OCR applied.")
                else:
                    st.success(f" Extracted `{q_file.name}` ({len(parsed_q.text.split())} words)")
                
                raw_paper_text = st.text_area("Extracted Exam Questions (Review or edit if needed):", value=parsed_q.text, height=130, key="q_extracted_edit")
            else:
                st.info("⬆️ Please upload your Question Paper PDF above to automatically detect questions and marks.")
        else:
            raw_paper_text = st.text_area(
                "Paste Exam Paper (with Q1, Q2, marks):",
                placeholder="Example:\nQ1. Explain your first question. [5 Marks]\n\nQ2. Explain your second question. [5 Marks]",
                height=130,
                key="q_typed_custom"
            )

        parsed_exam = extract_questions_from_text(raw_paper_text) if raw_paper_text.strip() else ExamQuestionPaper()
        
        if "excluded_questions" not in st.session_state:
            st.session_state["excluded_questions"] = set()

        # Filter out questions that the user manually deleted/excluded
        active_questions = [q for q in parsed_exam.questions if q.id not in st.session_state["excluded_questions"]]
        parsed_exam.questions = active_questions
        parsed_exam.total_exam_marks = sum(q.max_marks for q in active_questions)

        if parsed_exam.questions:
            col_m1, col_m2 = st.columns([3, 1])
            col_m1.markdown(f"**📌 Detected `{len(parsed_exam.questions)}` Question(s) — Total Exam Marks: `{parsed_exam.total_exam_marks}` pts**")
            if st.session_state["excluded_questions"]:
                if col_m2.button("🔄 Restore Removed", key="restore_del_qs_btn"):
                    st.session_state["excluded_questions"] = set()
                    st.rerun()

            with st.expander("📝 Review & Adjust Marks Allocation per Question", expanded=True):
                for idx, q in enumerate(parsed_exam.questions):
                    c_q1, c_q2, c_q3 = st.columns([3, 1, 0.4])
                    c_q1.write(f"**{q.title}:** {q.text[:65]}...")
                    q.max_marks = c_q2.number_input(f"{q.id} Marks", min_value=1.0, max_value=50.0, value=float(q.max_marks), step=1.0, key=f"m_{q.id}_{idx}")
                    if c_q3.button("🗑️", key=f"del_{q.id}_{idx}", help="Remove this question or instruction from grading"):
                        st.session_state["excluded_questions"].add(q.id)
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="card-header-tag">STEP 2 OF 3</div>
            <div class="card-title">Reference Answer / Model Solutions</div>
        """, unsafe_allow_html=True)
        
        ref_mode = st.radio(
            "Model Solution Source:",
            ["✨ Auto-Generate All Solutions with AI", "📄 Upload Full Answer Key PDF", "✍️ Type Custom Solutions Manually"],
            horizontal=True
        )
        
        if ref_mode == "✨ Auto-Generate All Solutions with AI":
            if st.button("🪄 Synthesize Authoritative Model Solutions with AI"):
                if parsed_exam.questions:
                    with st.spinner("AI is generating reference solutions per question..."):
                        for q in parsed_exam.questions:
                            q.reference_answer = rubric_engine.generate_reference_answer(q.text, q.max_marks)
                        st.session_state['exam_solutions'] = {q.id: q.reference_answer for q in parsed_exam.questions}
                else:
                    st.warning("Please provide a question paper first in Step 1.")
            
            sol_dict = st.session_state.get('exam_solutions', {})
            for q in parsed_exam.questions:
                q.reference_answer = sol_dict.get(q.id, rubric_engine.generate_reference_answer(q.text, q.max_marks))
                st.caption(f"**{q.title} Reference Solution:** *{q.reference_answer[:90]}...*")

        elif ref_mode == "📄 Upload Full Answer Key PDF":
            sol_file = st.file_uploader("Upload Complete Answer Key / Marking Scheme (PDF/TXT):", type=["pdf", "txt"], key="full_ans_pdf")
            if sol_file:
                parsed_sol_doc = parse_uploaded_file(sol_file, llm)
                mapped_solutions = split_student_answers_for_exam(parsed_sol_doc.text, parsed_exam.questions)
                st.success(f" Extracted solutions from `{sol_file.name}` mapped across {len(parsed_exam.questions)} question(s)!")
                
                with st.expander("👁️ Review & Edit Extracted Question Solutions", expanded=True):
                    for idx, q in enumerate(parsed_exam.questions):
                        default_sol = mapped_solutions.get(q.id, parsed_sol_doc.text)
                        q.reference_answer = st.text_area(f"Extracted Solution for {q.title}:", value=default_sol, height=80, key=f"sol_pdf_{q.id}_{idx}")
            else:
                st.info("Upload your teacher's marking scheme or full answer key PDF above.")

        else:
            for idx, q in enumerate(parsed_exam.questions):
                q.reference_answer = st.text_area(f"Reference Solution for {q.title}:", value=q.reference_answer or "", placeholder="Type reference answer here...", height=70, key=f"ref_{q.id}_{idx}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-tag">STEP 3 OF 3</div>
            <div class="card-title">Upload Student Solution Papers</div>
        """, unsafe_allow_html=True)
        
        student_files = st.file_uploader(
            "Upload Student Submission Files (PDF, PNG, JPG):",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
            help="Upload multi-page student exam answer sheets."
        )

        st.caption(f"📁 **Files Uploaded:** {len(student_files) if student_files else 0} student exam paper(s)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Prominent Start Grading Action Box (Always Visible)
        num_students = len(student_files) if student_files else 0
        num_questions = len(parsed_exam.questions) if parsed_exam.questions else 1

        btn_label = f"🚀 Start Automated Grading ({num_students} Student Paper{'s' if num_students != 1 else ''} • {num_questions} Question{'s' if num_questions != 1 else ''})"
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%); border: 1.5px solid #818CF8; border-radius: 12px; padding: 1.2rem; margin-top: 0.5rem; text-align: center;">
            <div style="font-weight: 700; color: #312E81; font-size: 1.05rem; margin-bottom: 0.5rem;">Ready to Grade?</div>
            <div style="color: #4338CA; font-size: 0.85rem; margin-bottom: 0.9rem;">Runs multi-model consensus (LLM + DeBERTa-v3 + MPNet) with character-level grounding.</div>
        """, unsafe_allow_html=True)
        
        start_grading_clicked = st.button(btn_label, type="primary", use_container_width=True, key="main_start_grade_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        if start_grading_clicked:
            if not student_files:
                st.error("⚠️ Please upload at least one Student Solution PDF/Image in Step 3 above to start grading.")
            elif not parsed_exam.questions or not raw_paper_text.strip():
                st.error("⚠️ Please provide your Question Paper in Step 1.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                status_text.text("Generating atomic rubrics for all exam questions...")
                exam_rubrics = {}
                for q in parsed_exam.questions:
                    ref_sol = q.reference_answer or rubric_engine.generate_reference_answer(q.text, q.max_marks)
                    exam_rubrics[q.id] = rubric_engine.generate_rubric(
                        question_id=q.id,
                        question_text=q.text,
                        reference_answer=ref_sol,
                        total_marks=q.max_marks
                    )

                exam_batch_results = []
                for s_idx, file_obj in enumerate(student_files):
                    status_text.text(f"Grading Student Paper {s_idx+1}/{len(student_files)}: {file_obj.name}...")
                    
                    parsed_doc = parse_uploaded_file(file_obj, llm)
                    mapped_answers = split_student_answers_for_exam(parsed_doc.text, parsed_exam.questions)
                    
                    student_exam_total = 0.0
                    student_exam_max = sum(q.max_marks for q in parsed_exam.questions)
                    q_reports = {}
                    conf_list = []

                    for q in parsed_exam.questions:
                        ans_chunk = mapped_answers.get(q.id, parsed_doc.text)
                        rep = confidence_engine.grade_full_answer(
                            student_id=100 + s_idx,
                            rubric=exam_rubrics[q.id],
                            student_answer=ans_chunk
                        )
                        q_reports[q.id] = rep
                        student_exam_total += rep.total_score
                        conf_list.append(rep.composite_confidence)

                    avg_conf = float(sum(conf_list) / max(1, len(conf_list)))
                    overall_route = "auto_accept" if avg_conf >= 0.80 else ("flag_for_spot_check" if avg_conf >= 0.50 else "requires_review")

                    # Automatically persist to Database so records appear in HITL Review Queue
                    try:
                        for q in parsed_exam.questions:
                            r_obj = exam_rubrics[q.id]
                            r_ver = db.query(RubricVersion).filter(RubricVersion.question_id == q.id).first()
                            if not r_ver:
                                r_ver = RubricVersion(
                                    question_id=q.id,
                                    version=1,
                                    schema_json=r_obj.model_dump_json(),
                                    is_active=True
                                )
                                db.add(r_ver)
                                db.commit()
                                db.refresh(r_ver)

                            ans_chunk = mapped_answers.get(q.id, parsed_doc.text)
                            rep = q_reports[q.id]
                            
                            sub = StudentSubmission(
                                question_id=q.id,
                                student_id=100 + s_idx,
                                answer_text=ans_chunk,
                                human_score_1=rep.total_score,
                                human_score_2=rep.total_score
                            )
                            db.add(sub)
                            db.commit()
                            db.refresh(sub)

                            for c_res in rep.criterion_results:
                                spans_json = json.dumps([s.model_dump() for s in c_res.segmentation.combined_evidence_spans])
                                g = GradingRecord(
                                    submission_id=sub.id,
                                    rubric_version_id=r_ver.id,
                                    criterion_id=c_res.criterion.id,
                                    evidence_spans_json=spans_json,
                                    tentative_score=c_res.score_result.points_awarded,
                                    max_points=c_res.criterion.points,
                                    final_score=c_res.score_result.points_awarded,
                                    confidence_score=c_res.confidence_score,
                                    routing_decision=c_res.routing,
                                    justification=c_res.score_result.justification,
                                    is_overridden=False
                                )
                                db.add(g)
                            db.commit()
                    except Exception as db_err:
                        print(f"DB persist notice: {db_err}")

                    exam_batch_results.append({
                        "Filename": file_obj.name,
                        "Type": "✏️ Handwritten/Image" if parsed_doc.is_handwritten_or_scanned else "📄 Digital PDF",
                        "OCR Legibility": f"{parsed_doc.ocr_confidence*100:.0f}%",
                        "Total Marks": f"{student_exam_total:.1f} / {student_exam_max:.1f}",
                        "Percentage": f"{(student_exam_total / max(1.0, student_exam_max)) * 100:.1f}%",
                        "Confidence": f"{avg_conf*100:.1f}%",
                        "Overall Status": overall_route,
                        "QuestionReports": q_reports,
                        "Questions": parsed_exam.questions,
                        "StudentAnswersMap": mapped_answers,
                        "ExtractedText": parsed_doc.text
                    })
                    progress_bar.progress((s_idx + 1) / len(student_files))

                status_text.text("✅ Full exam grading completed!")
                st.session_state['exam_batch_results'] = exam_batch_results

    if 'exam_batch_results' in st.session_state:
        st.markdown("---")
        results = st.session_state['exam_batch_results']
        
        # Executive KPI Header
        st.markdown("### 📊 Executive Examination Performance & Quality Metrics")
        
        total_subs = len(results)
        auto_accepted_cnt = sum(1 for r in results if r["Overall Status"] == "auto_accept")
        flagged_cnt = total_subs - auto_accepted_cnt
        avg_pct = sum(float(r["Percentage"].replace("%", "")) for r in results) / max(1, total_subs)
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">TOTAL SUBMISSIONS</div>
                <div class="kpi-value">{total_subs}</div>
                <div style="font-size:0.75rem; color:#64748B;">Processed Papers</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">CLASS AVERAGE</div>
                <div class="kpi-value">{avg_pct:.1f}%</div>
                <div style="font-size:0.75rem; color:#10B981;">Mean Batch Performance</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">AUTO-ACCEPTED</div>
                <div class="kpi-value">{auto_accepted_cnt} <span style="font-size:1rem; color:#64748B;">({(auto_accepted_cnt/max(1,total_subs))*100:.0f}%)</span></div>
                <div style="font-size:0.75rem; color:#10B981;">High Confidence (&ge; 80%)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">SPOT-CHECK QUEUE</div>
                <div class="kpi-value">{flagged_cnt}</div>
                <div style="font-size:0.75rem; color:#F59E0B;">Routed to HITL Review</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📋 Consolidated Grade Sheet")
        
        df_exam = pd.DataFrame([{
            "Student Paper": r["Filename"],
            "Format": r["Type"],
            "OCR Confidence": r["OCR Legibility"],
            "Total Score": r["Total Marks"],
            "Percentage": r["Percentage"],
            "Exam Confidence": r["Confidence"],
            "Routing Action": r["Overall Status"]
        } for r in results])

        st.dataframe(df_exam, use_container_width=True)

        csv_buf = io.StringIO()
        df_exam.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Complete Exam Scorecard (CSV)",
            data=csv_buf.getvalue(),
            file_name="Complete_Exam_Grading_Report.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("### 📝 Detailed Per-Question Answer Evaluation & Evidence Breakdown")
        st.markdown("Inspect each student's **extracted answer, evidence highlights, and itemized grading checkpoints** below:")

        for res in results:
            pct_val = float(res['Percentage'].replace("%", ""))
            badge_color = "#10B981" if pct_val >= 80 else ("#F59E0B" if pct_val >= 50 else "#EF4444")
            
            with st.expander(f"🎓 {res['Filename']} — Total Score: {res['Total Marks']} ({res['Percentage']})", expanded=True):
                st.markdown(f"""
                <div style="display:flex; gap:12px; align-items:center; margin-bottom:1rem; padding:10px 14px; background:#F8FAFC; border-radius:8px; border-left:4px solid {badge_color};">
                    <span style="font-weight:700; color:#0F172A; font-size:1.05rem;">Student Result Summary:</span>
                    <span style="background:{badge_color}; color:#FFFFFF; font-weight:700; padding:3px 10px; border-radius:12px; font-size:0.85rem;">
                        Score: {res['Total Marks']} ({res['Percentage']})
                    </span>
                    <span style="background:#EEF2FF; color:#4338CA; font-weight:600; padding:3px 10px; border-radius:12px; font-size:0.85rem;">
                        Confidence: {res['Confidence']}
                    </span>
                    <span style="background:#E2E8F0; color:#334155; font-weight:600; padding:3px 10px; border-radius:12px; font-size:0.85rem;">
                        Action: {res['Overall Status']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                for q in res["Questions"]:
                    q_rep = res["QuestionReports"][q.id]
                    q_score_pct = (q_rep.total_score / max(0.1, q.max_marks)) * 100
                    
                    st.markdown(f"""
                    <div style="background:#FFFFFF; border:1.5px solid #CBD5E1; border-radius:12px; padding:1.2rem; margin-bottom:1.5rem; box-shadow:0 3px 6px rgba(0,0,0,0.03);">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.8rem; border-bottom:1px solid #F1F5F9; padding-bottom:0.6rem;">
                            <div>
                                <span style="font-size:1.1rem; font-weight:700; color:#1E1B4B;">{q.title}: </span>
                                <span style="font-size:1.0rem; color:#334155; font-weight:500;">{q.text}</span>
                            </div>
                            <div style="text-align:right; min-width:140px;">
                                <span style="background:#EEF2FF; color:#3730A3; font-weight:700; padding:5px 12px; border-radius:8px; font-size:0.95rem;">
                                    {q_rep.total_score:.1f} / {q.max_marks:.1f} pts ({q_score_pct:.0f}%)
                                </span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Extract all cited evidence spans for this question
                    all_spans = []
                    for c_res in q_rep.criterion_results:
                        for span in c_res.segmentation.combined_evidence_spans:
                            if span.text.strip():
                                all_spans.append(span.text.strip())

                    # Actual student text extracted for this question from PDF
                    actual_student_text = res.get("StudentAnswersMap", {}).get(q.id, res.get("ExtractedText", ""))
                    if not actual_student_text.strip():
                        actual_student_text = res.get("ExtractedText", "[No text extracted from this page]")

                    highlighted_student_text = actual_student_text
                    for sp in all_spans:
                        if sp and sp in highlighted_student_text:
                            highlighted_student_text = highlighted_student_text.replace(
                                sp,
                                f'<span class="evidence-highlight">{sp}</span>'
                            )

                    # Side-by-Side 2 Columns: Left = Student Answer, Right = Rubric Checkpoints
                    c_ans, c_rubric = st.columns([1, 1])

                    with c_ans:
                        st.markdown("""
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-weight:700; color:#1E293B; font-size:0.95rem;">
                                📄 Student's Answer Sheet (Extracted from PDF)
                            </span>
                            <span style="font-size:0.75rem; background:#EEF2FF; color:#4338CA; padding:2px 8px; border-radius:6px; font-weight:600;">
                                Verbatim Extracted
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style="background:#F8FAFC; border:1.5px solid #E2E8F0; border-radius:8px; padding:14px; min-height:160px; line-height:1.6; font-size:0.92rem; color:#1E293B;">
                            {highlighted_student_text}
                        </div>
                        """, unsafe_allow_html=True)
                            
                        st.caption("🟡 *Yellow highlighted text represents verbatim evidence verified by AI.*")

                    with c_rubric:
                        st.markdown("""
                        <div style="font-weight:700; color:#1E293B; margin-bottom:6px; font-size:0.95rem;">
                            📋 Rubric Checkpoints & AI Evaluation
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for c_idx, c_res in enumerate(q_rep.criterion_results, 1):
                            c = c_res.criterion
                            s_res = c_res.score_result
                            seg = c_res.segmentation

                            c_badge = "✅ Full Credit" if s_res.points_awarded == c.points else ("⚠️ Partial Credit" if s_res.points_awarded > 0 else "❌ Zero Credit")
                            badge_bg = "#ECFDF5" if s_res.points_awarded == c.points else ("#FFFBEB" if s_res.points_awarded > 0 else "#FEF2F2")
                            badge_txt = "#065F46" if s_res.points_awarded == c.points else ("#92400E" if s_res.points_awarded > 0 else "#991B1B")

                            st.markdown(f"""
                            <div style="background:{badge_bg}; border:1px solid #CBD5E1; border-radius:8px; padding:10px 12px; margin-bottom:8px;">
                                <div style="display:flex; justify-content:space-between; align-items:center; font-weight:600; font-size:0.9rem; color:#0F172A;">
                                    <span>Checkpoint {c_idx}: {c.description}</span>
                                    <span style="color:{badge_txt}; font-weight:700;">{s_res.points_awarded:.1f} / {c.points:.1f} pts ({c_badge})</span>
                                </div>
                                <div style="font-size:0.83rem; color:#475569; margin-top:3px;">
                                    <b>Condition:</b> {c.satisfaction_condition}
                                </div>
                                <div style="font-size:0.85rem; color:#1E293B; margin-top:4px; line-height:1.4;">
                                    <b>AI Justification:</b> {s_res.justification}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Backend consensus pills
                            b1 = "✅" if seg.llm_result.evidence_found else "❌"
                            b2 = "✅" if seg.deberta_result.evidence_found else "❌"
                            b3 = "✅" if seg.mpnet_result.evidence_found else "❌"
                            st.caption(f"🤝 **Consensus:** LLM ({b1}) | DeBERTa-v3 ({b2}) | MPNet ({b3}) — Jaccard Overlap: `{seg.span_overlap_agreement:.2f}`")

                    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: LIVE INTERACTIVE GRADING CONSOLE
# -------------------------------------------------------------
elif nav_choice == "⚡ Live Interactive Grading Console":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">⚡ Live Interactive Grading Console</div>
        <div class="hero-subtitle">Inspect the 4-stage pipeline step-by-step: Rubric Generation ➔ 3-Backend Consensus ➔ Strict Scorer ➔ Confidence Routing.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-header-tag">BENCHMARK PROMPT</div>
            <div class="card-title">1. Select Question & Answer</div>
        """, unsafe_allow_html=True)
        
        source_mode = st.radio("Source Mode:", ["✍️ Custom Freeform Question", "📚 Benchmark Datasets"], horizontal=True)
        
        if source_mode == "✍️ Custom Freeform Question":
            custom_q = st.text_area("Question Prompt:", value="Explain why photosynthesis is essential for maintaining atmospheric balance.", height=80)
            custom_max_marks = st.number_input("Max Marks:", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
            custom_ref = st.text_area("Model Reference Answer (Optional):", value="Photosynthesis takes in CO2 and releases O2, keeping oxygen and carbon dioxide levels balanced.", height=80)
            student_answer = st.text_area("Student's Submitted Answer:", height=120, placeholder="Type student answer to test...")
            student_id = 101
            dataset = {
                "question_id": "Custom_Q1",
                "question_text": custom_q,
                "total_marks": custom_max_marks,
                "reference_answer": custom_ref
            }
            grade_btn = st.button("🚀 Run Multi-Model Grading Pipeline", type="primary", use_container_width=True)
        else:
            dataset_choice = st.selectbox("Exam Set", ["ASAP Set 1: Science (Mass Conservation)", "ASAP Set 2: Biology (Cellular Function)"])
            set_id = "1" if "Set 1" in dataset_choice else "2"
            dataset = load_asap_dataset(set_id)

            if dataset:
                st.info(f"**Prompt:** {dataset['question_text']}")
                sample_answers = {f"Student #{r['id']} (Human Score: {r['human_score_1']})": r["student_answer"] for r in dataset["sample_records"]}
                sample_answers["[Custom Freeform Answer]"] = ""
                
                selected_student_key = st.selectbox("Choose sample response or custom:", list(sample_answers.keys()))
                if selected_student_key == "[Custom Freeform Answer]":
                    student_answer = st.text_area("Enter Student's Answer:", height=150, placeholder="Type subjective answer here...")
                    student_id = 999
                else:
                    student_answer = st.text_area("Student's Answer:", value=sample_answers[selected_student_key], height=150)
                    student_id = int(selected_student_key.split("#")[1].split(" ")[0])

                grade_btn = st.button("🚀 Run Multi-Model Grading Pipeline", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if dataset and (grade_btn or 'last_graded' in st.session_state):
            with st.spinner("Executing 4-stage grading pipeline..."):
                rubric = rubric_engine.generate_rubric(
                    question_id=dataset["question_id"],
                    question_text=dataset["question_text"],
                    reference_answer=dataset["reference_answer"],
                    total_marks=dataset["total_marks"]
                )

                report = confidence_engine.grade_full_answer(
                    student_id=student_id,
                    rubric=rubric,
                    student_answer=student_answer
                )
                st.session_state['last_graded'] = report

                # Automatically persist to Database so it immediately connects to HITL Review Queue
                try:
                    r_ver = db.query(RubricVersion).filter(RubricVersion.question_id == dataset["question_id"]).first()
                    if not r_ver:
                        r_ver = RubricVersion(
                            question_id=dataset["question_id"],
                            version=1,
                            schema_json=rubric.model_dump_json(),
                            is_active=True
                        )
                        db.add(r_ver)
                        db.commit()
                        db.refresh(r_ver)

                    sub = StudentSubmission(
                        question_id=dataset["question_id"],
                        student_id=student_id,
                        answer_text=student_answer,
                        human_score_1=report.total_score,
                        human_score_2=report.total_score
                    )
                    db.add(sub)
                    db.commit()
                    db.refresh(sub)

                    for c_res in report.criterion_results:
                        spans_json = json.dumps([s.model_dump() for s in c_res.segmentation.combined_evidence_spans])
                        g = GradingRecord(
                            submission_id=sub.id,
                            rubric_version_id=r_ver.id,
                            criterion_id=c_res.criterion.id,
                            evidence_spans_json=spans_json,
                            tentative_score=c_res.score_result.points_awarded,
                            max_points=c_res.criterion.points,
                            final_score=c_res.score_result.points_awarded,
                            confidence_score=c_res.confidence_score,
                            routing_decision=c_res.routing,
                            justification=c_res.score_result.justification,
                            is_overridden=False
                        )
                        db.add(g)
                    db.commit()
                except Exception as db_err:
                    print(f"Tab 2 DB sync notice: {db_err}")

            if report.overall_routing == "auto_accept":
                badge_html = '<span class="badge-auto">✅ AUTO-ACCEPTED</span>'
            elif report.overall_routing == "flag_for_spot_check":
                badge_html = '<span class="badge-spot">⚠️ FLAGGED FOR SPOT-CHECK</span>'
            else:
                badge_html = '<span class="badge-review">🚨 REQUIRES HUMAN REVIEW</span>'

            # Metric KPIs
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-num">{report.total_score} / {report.max_total_score}</div>
                    <div class="kpi-label">Awarded Score</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col2:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-num">{report.composite_confidence * 100:.1f}%</div>
                    <div class="kpi-label">Ensemble Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with kpi_col3:
                st.markdown(f"""
                <div class="kpi-box">
                    <div style="margin-top:6px;">{badge_html}</div>
                    <div class="kpi-label">Routing Decision</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### 🔍 Evidence-Grounded Criterion Breakdown")
            for idx, crit_res in enumerate(report.criterion_results, 1):
                c = crit_res.criterion
                s_res = crit_res.score_result
                seg = crit_res.segmentation

                with st.expander(f"Criterion {idx}: {c.description} ({s_res.points_awarded}/{c.points} pts)", expanded=True):
                    highlighted_answer = student_answer
                    for span in seg.combined_evidence_spans:
                        if span.text in highlighted_answer:
                            highlighted_answer = highlighted_answer.replace(
                                span.text,
                                f'<span class="evidence-highlight">{span.text}</span>'
                            )
                    
                    st.markdown(f"**Student Answer (with cited evidence highlighted):**")
                    st.markdown(f"> {highlighted_answer}", unsafe_allow_html=True)
                    st.markdown(f"**AI Justification:** *{s_res.justification}*")
                    st.caption(f"📐 **Jaccard Overlap:** `{seg.span_overlap_agreement:.2f}` | **Criterion Confidence:** `{crit_res.confidence_score:.2f}`")

# -------------------------------------------------------------
# TAB 3: INSTRUCTOR REVIEW QUEUE (HITL)
# -------------------------------------------------------------
elif nav_choice == "👨‍🏫 Instructor Review Queue (HITL)":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">👨‍🏫 Instructor Review Queue (HITL)</div>
        <div class="hero-subtitle">Review ambiguous answers flagged by the confidence engine. Humans only review the 10–20% flagged edge cases.</div>
    </div>
    """, unsafe_allow_html=True)

    col_h1, col_h2 = st.columns([3, 1])
    col_h1.write("")
    if col_h2.button("🗑️ Wipe All Database Records", key="wipe_db_btn", help="Clear all stored student submissions and grading records"):
        db.query(GradingRecord).delete()
        db.query(StudentSubmission).delete()
        db.query(RubricVersion).delete()
        db.query(AuditTrail).delete()
        db.commit()
        st.success("All grading records wiped clean!")
        st.rerun()

    records = db.query(GradingRecord).all()
    if not records:
        st.info("The Review Queue is currently empty. As you grade student exam papers in Batch PDF Grading or evaluate answers in Live Grading Console, all graded submissions will automatically appear here for your review and score audit.")
    else:
        filter_route = st.selectbox("Filter by Action:", ["All Records", "requires_review", "flag_for_spot_check", "auto_accept"])
        query = db.query(GradingRecord)
        if filter_route != "All Records":
            query = query.filter(GradingRecord.routing_decision == filter_route)
        
        filtered_records = query.all()
        # Group records by Student Submission (Student ID + Question)
        sub_ids = list(dict.fromkeys(r.submission_id for r in filtered_records))
        st.write(f"Showing **{len(sub_ids)}** Student Questions (**{len(filtered_records)}** total rubric checkpoints).")

        for s_id in sub_ids:
            sub = db.query(StudentSubmission).filter(StudentSubmission.id == s_id).first()
            sub_recs = [r for r in filtered_records if r.submission_id == s_id]
            q_total_score = sum(r.final_score for r in sub_recs)
            q_max_score = sum(r.max_points for r in sub_recs)

            with st.container():
                st.markdown(f"### 🎓 Student ID: `{sub.student_id if sub else 'N/A'}` | Question: `{sub.question_id if sub else 'N/A'}` | Score: `{q_total_score:.1f} / {q_max_score:.1f} pts`")
                st.markdown(f"**Submitted Answer:** {sub.answer_text if sub else 'N/A'}")
                
                # Checkpoints under this question
                for g_rec in sub_recs:
                    with st.expander(f"📌 Checkpoint: {g_rec.criterion_id} — Score: {g_rec.final_score}/{g_rec.max_points} pts ({g_rec.routing_decision})", expanded=True):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.write(f"**AI Justification:** *{g_rec.justification}*")
                            st.caption(f"Confidence: `{g_rec.confidence_score:.2f}` | Tentative: `{g_rec.tentative_score} pts`")
                        with c2:
                            new_score = st.number_input(f"Score ({g_rec.criterion_id})", min_value=0.0, max_value=float(g_rec.max_points), value=float(g_rec.final_score), step=0.5, key=f"num_{g_rec.id}")
                            btn_c1, btn_c2 = st.columns(2)
                            with btn_c1:
                                if st.button("✅ Accept", key=f"acc_{g_rec.id}"):
                                    g_rec.final_score = g_rec.tentative_score
                                    g_rec.is_overridden = False
                                    db.commit()
                                    st.success("Accepted.")
                            with btn_c2:
                                if st.button("✏️ Save", key=f"ovr_{g_rec.id}"):
                                    g_rec.final_score = new_score
                                    g_rec.is_overridden = True
                                    db.commit()
                                    st.success("Saved.")
                st.markdown("---")

# -------------------------------------------------------------
# TAB 4: RUBRIC REFINEMENT & RE-SCORING
# -------------------------------------------------------------
elif nav_choice == "🔄 Versioned Rubrics & Re-scoring":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🔄 Versioned Rubrics & Retroactive Re-scoring</div>
        <div class="hero-subtitle">Select any exam question to inspect active rubric versions, propose AI amendments for newly identified student points, and automatically re-score all cohort submissions.</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Fetch all distinct active rubrics across all questions
    all_active_rubrics = db.query(RubricVersion).filter(RubricVersion.is_active == True).all()

    if all_active_rubrics:
        q_options = [r.question_id for r in all_active_rubrics]
        
        # Check if a question was pre-selected from HITL "Flag Rubric Gap"
        flagged_info = st.session_state.get('flagged_gap', {})
        flagged_qid = flagged_info.get('question_id', q_options[0])
        default_idx = q_options.index(flagged_qid) if flagged_qid in q_options else 0
        
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_qid = st.selectbox(
                "📌 Select Exam Question to Refine or Re-score:",
                options=q_options,
                index=default_idx,
                format_func=lambda q: f"Exam Question: {q}"
            )
        with col_sel2:
            st.caption(f"Total Available Questions: **{len(q_options)}**")

        active_rubric_rec = db.query(RubricVersion).filter(
            RubricVersion.question_id == selected_qid,
            RubricVersion.is_active == True
        ).first()
    if active_rubric_rec:
        st.markdown(f"#### Active Rubric: `{active_rubric_rec.question_id}` (Version {active_rubric_rec.version})")
        curr_schema = RubricSchema.model_validate_json(active_rubric_rec.schema_json)
        
        with st.expander("View Active Rubric Checkpoints", expanded=False):
            st.json(curr_schema.model_dump())

        st.subheader("Add a New Valid Point to Rubric")
        st.caption("If a student wrote a valid point that wasn't in your answer key, paste that sentence below to award marks for it.")
        
        excerpt_val = st.session_state.get('flagged_gap', {}).get('excerpt', "")
        inst_note = st.text_input("Teacher's Note (What should earn marks?):", placeholder="e.g. Award credit if student mentions specific key concept...")
        student_excerpt = st.text_area("Student's Sentence / Quote (Copy-Pasted from Answer):", value=excerpt_val)

        if st.button("✨ Propose New Point with AI (Stage 6)"):
            with st.spinner("Analyzing rubric gap with LLM..."):
                refinement = rubric_engine.propose_refinement(
                    current_rubric=curr_schema,
                    instructor_flag_text=inst_note,
                    student_answer_excerpt=student_excerpt
                )
                st.session_state['pending_refinement'] = refinement
                st.session_state['curr_schema'] = curr_schema

        if 'pending_refinement' in st.session_state:
            ref = st.session_state['pending_refinement']
            st.success(f"**Proposed Change:** `{ref.proposed_change}`\n\n**Rationale:** {ref.rationale}")
            st.json(ref.details)

            if st.button("🚀 Approve & Trigger Retroactive Re-Score Queue (Stage 7)", type="primary"):
                updated_criteria = list(curr_schema.criteria)
                new_crit_data = ref.details.get("new_criterion", {})
                if new_crit_data:
                    updated_criteria.append(RubricCriterion(
                        id=new_crit_data.get("id", "crit_bonus"),
                        description=new_crit_data.get("description", "Bonus concept"),
                        points=float(new_crit_data.get("points", 0.5)),
                        satisfaction_condition=new_crit_data.get("satisfaction_condition", "Explicitly names concept"),
                        keywords_or_concepts=["CO2", "conservation of mass"]
                    ))

                new_rubric_schema = RubricSchema(
                    question_id=curr_schema.question_id,
                    total_marks=curr_schema.total_marks + 0.5,
                    criteria=updated_criteria
                )

                queue = RescoringQueue(db, confidence_engine)
                res = queue.apply_rubric_refinement(
                    question_id=curr_schema.question_id,
                    new_rubric_schema=new_rubric_schema,
                    refinement_result=ref
                )

                st.balloons()
                st.success(f"✅ Upgraded to Rubric Version {res['new_version']}! Retroactively re-evaluated {res['rescored_count']} submissions.")
                st.dataframe(pd.DataFrame(res["details"]))
    else:
        st.info("No active rubric in DB. Grade a submission first.")

# -------------------------------------------------------------
# TAB 5: HYBRID RAG & SQL ANALYTICS
# -------------------------------------------------------------
elif nav_choice == "💬 Hybrid RAG & SQL Analytics":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">💬 Hybrid RAG & SQL Analytics Assistant</div>
        <div class="hero-subtitle">Dispatches statistical cohort questions to SQL and qualitative grading queries to ChromaDB vector search.</div>
    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input("Ask a question about grading records or class statistics:", placeholder="e.g. 'What is the average score and auto-accept rate?' or 'Why did Student #104 get 0 marks?'")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        if st.button("Sample: 'What is the average score and auto-accept distribution?'"):
            user_query = "What is the average score and auto-accept distribution?"
    with col_q2:
        if st.button("Sample: 'Why did Student #104 receive a score of 0?'"):
            user_query = "Why did Student #104 receive a score of 0?"

    if user_query:
        with st.spinner("Routing query and generating response..."):
            ans = rag_engine.query(user_query)

        st.markdown(f"**Query Route Detected:** `{ans['route'].upper()}`")
        st.markdown(ans["response"])
        with st.expander("Retrieved Sources / Context Metadata", expanded=False):
            st.json(ans["sources"])

