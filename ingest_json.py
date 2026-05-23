import json
import logging
from vector_db import VectorDB
from chunker import semantic_chunk_document
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_json_data():
    logger.info("Loading translation_cache.json...")
    try:
        with open('data/translation_cache.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error("data/translation_cache.json not found!")
        return

    chunks_to_embed = []
    logger.info(f"Loaded {len(data)} translation entries. Chunking...")
    
    for ja_text, en_text in data.items():
        # Skip extremely short entries (less than 150 characters) which are just page titles or navigation breadcrumbs
        if len(en_text.strip()) < 150:
            continue
            
        doc = {
            "url": "local-json-cache",
            "title": "MMI Translated Content",
            "content_ja": ja_text,
            "content_en": en_text,
            "headings": []
        }
        chunks = semantic_chunk_document(doc, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
        chunks_to_embed.extend(chunks)

    logger.info(f"Generated {len(chunks_to_embed)} chunks. Adding to vector db...")
    
    vector_db = VectorDB()
    vector_db.add_chunks(chunks_to_embed)
    
    logger.info("Successfully ingested JSON data into Vector DB!")

if __name__ == "__main__":
    ingest_json_data()
