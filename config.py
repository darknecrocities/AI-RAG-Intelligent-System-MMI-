import os

# Base URL for the target website to crawl
BASE_URL = "https://www.mmi-sc.co.jp/"

# Hugging Face Space Settings
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "Darknecrocities/mmi2")

# Lightweight mode: skip loading PyTorch/sentence-transformers locally,
# use HuggingFace Inference API for embeddings instead (for 512MB RAM hosts like Render free tier)
LIGHTWEIGHT_MODE = os.getenv("LIGHTWEIGHT_MODE", "false").lower() == "true"

# Embeddings and Reranking models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Disable Cross Encoder by default to avoid Out of Memory (OOM) on 512MB RAM instances
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "")

# Chunking settings
CHUNK_SIZE = 600       # target token length for chunks
CHUNK_OVERLAP = 100    # overlap token count

# Data persistence directories
DATA_DIR = os.getenv("DATA_DIR", "./data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "faiss_index")
CRAWL_CACHE_PATH = os.path.join(DATA_DIR, "crawl_cache.json")
TRANSLATION_CACHE_PATH = os.path.join(DATA_DIR, "translation_cache.json")

# Gemini API Key (for cloud low-memory mode)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# Redis configuration (falls back to memory if connection fails)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# API Server Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
