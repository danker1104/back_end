import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# 모델 설정
MODEL_NAME = "LGAI-EXAONE/EXAONE-4.0-1.2B"
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
