import os

# Base URL for the target website to crawl
BASE_URL = "https://www.mmi-sc.co.jp/"

# Ollama local connection settings
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# Embeddings and Reranking models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

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
