import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# LLM Providers and Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq" if GROQ_API_KEY else ("openai" if OPENAI_API_KEY else "mock"))

# Model Families
RUBRIC_MODEL = os.getenv("RUBRIC_MODEL", "openai/gpt-oss-20b" if LLM_PROVIDER == "groq" else "gpt-4o")
SCORING_MODEL = os.getenv("SCORING_MODEL", "openai/gpt-oss-20b" if LLM_PROVIDER == "groq" else "gpt-4o-mini")

# Small Local NLP Models
DEBERTA_QA_MODEL = "deepset/deberta-v3-base-squad2"
MPNET_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Confidence Routing Thresholds
CONFIDENCE_AUTO_ACCEPT = float(os.getenv("CONFIDENCE_AUTO_ACCEPT", "0.80"))
CONFIDENCE_SPOT_CHECK = float(os.getenv("CONFIDENCE_SPOT_CHECK", "0.50"))

# Database & Storage
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/grading_system.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))
