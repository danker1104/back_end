import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path=None):
        if path is None:
            return False
        try:
            with open(path, encoding="utf-8") as env_file:
                for line in env_file:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    name, value = line.split("=", 1)
                    name = name.strip()
                    value = value.strip().strip('"').strip("'")
                    if name and name not in os.environ:
                        os.environ[name] = value
            return True
        except FileNotFoundError:
            return False

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Gemini API 설정
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE_URL = os.getenv("GEMINI_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
MODEL_CACHE_DIR = PROJECT_ROOT / "model"
DEFAULT_MAX_LENGTH = 1024

# 장치 설정
DEVICE = "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu"

# Hugging Face 설정
HF_TOKEN = os.getenv("HF_TOKEN", "")

# 출력/저장 경로
OUTPUT_DIR = PROJECT_ROOT / "output"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# RAG 관련 기본 설정
TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32

# 디렉터리 생성
for directory in [MODEL_CACHE_DIR, OUTPUT_DIR, VECTOR_DB_DIR, EMBEDDINGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
