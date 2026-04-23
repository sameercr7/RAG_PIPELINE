import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("BACKUP_DATABASE", "up_police_matrix_dev_01")

HF_API_TOKEN  = os.getenv("HF_API_TOKEN", "")          # HuggingFace token (free)
HF_MODEL      = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")  # free HF model

# Local LLM settings
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "False").lower() == "true"
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "hf.co/mistralai/Mistral-7B-Instruct-v0.2")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = "police_matrix"

EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, lightweight, runs locally
INGEST_BATCH_SIZE = 50000           # Reduced for incremental ingestion
TOP_K = 10                         # top chunks to retrieve

# Persistent Admin State
STATE_FILE = os.path.join(os.getcwd(), "status_state.json")