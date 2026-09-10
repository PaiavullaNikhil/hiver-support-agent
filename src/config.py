from pathlib import Path
import os

from dotenv import load_dotenv


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "twcs"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"

RAW_DATA_PATH = RAW_DATA_DIR / "twcs.csv"
SAMPLE_DATA_PATH = RAW_DATA_DIR / "sample.csv"

CLEAN_APPLE_PATH = PROCESSED_DATA_DIR / "apple_support_clean.csv"
GOLDEN_SET_PATH = EVALUATION_DIR / "apple_support_golden_200.csv"


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

RANDOM_STATE = 2026


# ---------------------------------------------------------
# Intent classifier
# ---------------------------------------------------------

INTENT_MARGIN_THRESHOLD = 0.30


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

RETRIEVAL_MODEL_NAME = "all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 5
RETRIEVAL_SIMILARITY_THRESHOLD = 0.65


# ---------------------------------------------------------
# Gemini
# ---------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3.5-flash"

LLM_EVALUATION_PATH = (
    EVALUATION_DIR / "apple_support_llm_judge_30.csv"
)

HUMAN_REVIEW_PATH = (
    EVALUATION_DIR / "apple_support_human_review_10.csv"
)