import os
import json
import logging
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        self.model_name = config.EMBEDDING_MODEL
        self.db_path = config.VECTOR_DB_PATH
        
        logger.info(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        self.index = None
        self.metadata = []
        
        self.load_index()

    def _get_index_file(self) -> str:
        return os.path.join(self.db_path, "index.faiss")

    def _get_metadata_file(self) -> str:
        return os.path.join(self.db_path, "metadata.json")

    def load_index(self):
        """
        Loads the FAISS index and metadata map from disk if they exist.
        Otherwise, creates a new IndexFlatIP (Inner Product / Cosine Similarity when normalized).
        """
        os.makedirs(self.db_path, exist_ok=True)
        index_file = self._get_index_file()
        metadata_file = self._get_metadata_file()

        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded FAISS index from disk. Contains {self.index.ntotal} vectors.")
                return
            except Exception as e:
                logger.error(f"Failed to load FAISS index from disk: {e}. Reinitializing.")

        # Create new IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        logger.info("Initialized a new empty FAISS index (Flat Inner Product).")

    def save_index(self):
        """
        Saves the FAISS index binary and metadata JSON map to disk.
        """
        os.makedirs(self.db_path, exist_ok=True)
        index_file = self._get_index_file()
        metadata_file = self._get_metadata_file()

        try:
            faiss.write_index(self.index, index_file)
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors to disk.")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def add_chunks(self, chunks: List[Dict]):
        """
        Takes a list of chunk dicts, generates embeddings, normalizes them, 
        adds them to the FAISS index, and saves to disk.
        """
        if not chunks:
            return

        texts = [chunk["text_en"] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize vectors for cosine similarity (Inner Product of normalized vectors = Cosine Similarity)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype("float32"))
        
        # Store metadata
        for idx, chunk in enumerate(chunks):
            # Extract and store metadata mapping
            meta = {
                "url": chunk.get("url", ""),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "text_en": chunk.get("text_en", ""),
                "text_ja": chunk.get("text_ja", ""),
                "content_en_raw": chunk.get("content_en_raw", "")
            }
            self.metadata.append(meta)
            
        self.save_index()

    def clear(self):
        """
        Clears index and metadata completely.
        """
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.save_index()
        logger.info("Cleared FAISS index and metadata.")

    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        filters: Dict = None
    ) -> List[Dict]:
        """
        Searches the FAISS index. Computes cosine similarity, applies metadata filters,
        and returns list of results sorted by similarity.
        """
        if not self.index or self.index.ntotal == 0:
            return []

        # Generate query embedding
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        # Search FAISS index
        scores, indices = self.index.search(query_vector.astype("float32"), top_k * 3) # Over-fetch for filtering
        
        results = []
        scores = scores[0]
        indices = indices[0]
        
        for idx, score in zip(indices, scores):
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx]
            
            # Apply metadata filters if provided
            if filters:
                match = True
                for key, val in filters.items():
                    if meta.get(key) != val:
                        match = False
                        break
                if not match:
                    continue

            results.append({
                "score": float(score),
                "url": meta["url"],
                "title": meta["title"],
                "section": meta["section"],
                "text_en": meta["text_en"],
                "text_ja": meta["text_ja"],
                "content_en_raw": meta["content_en_raw"]
            })
            
            if len(results) >= top_k:
                break
                
        return results
