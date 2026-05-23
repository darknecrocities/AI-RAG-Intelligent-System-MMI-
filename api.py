import json
import logging
import time
from typing import List, Dict, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from crawler import WebCrawler
from extractor import extract_content
from translator import TranslationEngine
from chunker import semantic_chunk_document
from vector_db import VectorDB
from retriever import HybridRetriever
from llm import HuggingFaceEngine
from cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MMI-KNOWLEDGE RAG SYSTEM",
    description="Production-grade local RAG chatbot for MMI-SC website content",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger.info("Initializing RAG System components...")
cache_manager = CacheManager()
translation_engine = TranslationEngine()
vector_db = VectorDB()
retriever = HybridRetriever(vector_db)
llm_engine = HuggingFaceEngine()
crawler = WebCrawler()

# Keep track of ingestion state
existing_chunks = vector_db.index.ntotal if vector_db.index else 0
existing_pages = len(crawler.cache) if crawler.cache else 0

ingestion_status = {
    "status": "completed" if existing_chunks > 0 else "idle",
    "pages_processed": existing_pages,
    "chunks_added": existing_chunks,
    "last_error": None,
    "last_ingested": "Previously loaded" if existing_chunks > 0 else None
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_query: str
    history: Optional[List[ChatMessage]] = None
    stream: bool = False
    force_recrawl: bool = False # bypass cache if needed

class IngestRequest(BaseModel):
    force_recrawl: bool = False
    clear_index: bool = False

def run_ingestion_sync(force_recrawl: bool = False, clear_index: bool = False):
    """
    Synchronous helper to run the crawling and embedding pipeline.
    """
    global ingestion_status
    ingestion_status["status"] = "ingesting"
    ingestion_status["pages_processed"] = 0
    ingestion_status["chunks_added"] = 0
    ingestion_status["last_error"] = None
    
    start_time = time.time()
    try:
        if clear_index:
            logger.info("Clearing FAISS Vector Database index...")
            vector_db.clear()
            
        # 1. Crawl pages
        logger.info("Starting website crawling...")
        crawl_results = crawler.crawl(force_recrawl=force_recrawl)
        
        # 2. Extract, translate, and chunk only new or updated pages
        chunks_to_embed = []
        pages_count = 0
        
        for url, page in crawl_results.items():
            # If the page was newly fetched/updated, or we are clearing/rebuilding the index
            if page.get("updated") or clear_index:
                logger.info(f"Processing content for {url}...")
                
                # Content extraction
                extracted = extract_content(page["html"], url)
                
                # Language detection and translation
                original_text = extracted["content"]
                title = extracted["title"]
                
                # Process Japanese translation
                if translation_engine.is_japanese(original_text):
                    logger.info(f"Translating Japanese content for {url}...")
                    translated_text = translation_engine.translate_text(original_text)
                    translated_title = translation_engine.translate_text(title)
                else:
                    translated_text = original_text
                    translated_title = title
                    
                doc = {
                    "url": url,
                    "title": title,
                    "content_ja": original_text,
                    "content_en": translated_text,
                    "headings": extracted["headings"]
                }
                
                # Chunk document
                chunks = semantic_chunk_document(
                    document=doc,
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP
                )
                chunks_to_embed.extend(chunks)
                pages_count += 1
                
                ingestion_status["pages_processed"] = pages_count
                
        # 3. Embedding generation and database storage
        if chunks_to_embed:
            logger.info(f"Adding {len(chunks_to_embed)} new chunks to FAISS vector store...")
            vector_db.add_chunks(chunks_to_embed)
            ingestion_status["chunks_added"] = len(chunks_to_embed)
        else:
            logger.info("No pages were modified or added. Vector store is up to date.")
            
        duration = time.time() - start_time
        logger.info(f"Ingestion completed successfully in {duration:.2f} seconds.")
        
        ingestion_status["status"] = "completed"
        ingestion_status["last_ingested"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "status": "success",
            "pages_processed": pages_count,
            "chunks_added": len(chunks_to_embed),
            "duration_seconds": round(duration, 2)
        }
    except Exception as e:
        logger.exception("Ingestion failed")
        ingestion_status["status"] = "failed"
        ingestion_status["last_error"] = str(e)
        raise e

@app.post("/ingest")
def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers website crawling and updates the FAISS vector database.
    Can run synchronously or asynchronously in the background.
    """
    if ingestion_status["status"] == "ingesting":
        return {"status": "error", "message": "An ingestion job is already running."}
        
    background_tasks.add_task(
        run_ingestion_sync, 
        force_recrawl=request.force_recrawl, 
        clear_index=request.clear_index
    )
    
    return {
        "status": "started",
        "message": "Ingestion job started in the background."
    }

@app.get("/ingest/status")
def get_ingest_status():
    """
    Returns current status of the background crawl and ingest job.
    """
    return ingestion_status

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Answers questions using retrieved website knowledge.
    Supports in-memory/Redis query caching and real-time Server-Sent Events (SSE) streaming.
    """
    query = request.user_query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Cache lookup (Skip if streaming is requested, as streaming cannot be cached directly)
    cache_key = f"rag_cache:{query.lower()}"
    if not request.stream and not request.force_recrawl:
        cached_res = cache_manager.get(cache_key)
        if cached_res:
            logger.info(f"Serving cached response for: '{query}'")
            return cached_res

    # 2. Smart Query Rewriting
    optimized_query = llm_engine.rewrite_query(query)
    
    # 3. Retrieve relevant chunks
    logger.info(f"Searching vector database for: '{optimized_query}'...")
    retrieved_chunks = retriever.retrieve(optimized_query, top_k=5)
    
    # 4. Context Compression
    if retrieved_chunks:
        compressed_context = llm_engine.compress_context(retrieved_chunks)
    else:
        compressed_context = ""
    
    # Convert history models to raw dicts for prompt builder
    history_dicts = []
    if request.history:
        history_dicts = [h.dict() for h in request.history]

    # 5. Build prompt
    prompt = llm_engine.build_prompt(compressed_context, query, history_dicts)
    
    # Format list of citations/sources
    sources = []
    seen_urls = set()
    for chunk in retrieved_chunks:
        url = chunk["url"]
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "title": chunk["title"],
                "section": chunk["section"],
                "score": chunk["score"],
                "snippet": chunk["content_en_raw"][:250] + "..." # Include a short preview snippet
            })

    # 6. Streaming Mode
    if request.stream:
        def event_generator():
            try:
                # Stream the answer tokens
                for token in llm_engine.generate_stream(prompt):
                    yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Stream sources at the end of the SSE stream
                yield f"data: {json.dumps({'sources': sources})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.exception("Error in streaming response")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 7. Standard Mode
    logger.info("Generating response synchronously...")
    response = llm_engine.generate(prompt)
    
    result = {
        "answer": response["text"],
        "sources": sources
    }
    
    # Cache the generated response
    cache_manager.set(cache_key, result, ttl=3600)
    
    return result

@app.get("/health")
def health_check():
    """
    System status and database size.
    """
    hf_ok = llm_engine._check_hf_connection()
    redis_ok = cache_manager.redis_available
    vector_count = vector_db.index.ntotal if vector_db.index else 0
    crawl_cache_count = len(crawler.cache)
    
    return {
        "status": "healthy" if hf_ok else "degraded",
        "ollama_connected": hf_ok,
        "redis_connected": redis_ok,
        "vector_store_chunks": vector_count,
        "crawled_pages": crawl_cache_count,
        "timestamp": time.time()
    }

# Ensure static files directory exists
import os
os.makedirs("static", exist_ok=True)

# Mount the static files folder to serve the dashboard UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")
