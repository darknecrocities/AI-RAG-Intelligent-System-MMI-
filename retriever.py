import logging
import re
from typing import List, Dict
import config

logger = logging.getLogger(__name__)

# Conditionally import CrossEncoder (only in full mode)
CrossEncoder = None
if not config.LIGHTWEIGHT_MODE:
    try:
        from sentence_transformers import CrossEncoder as _CE
        CrossEncoder = _CE
    except ImportError:
        logger.warning("sentence-transformers not installed. CrossEncoder reranking disabled.")
class HybridRetriever:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.cross_encoder_name = config.CROSS_ENCODER_MODEL
        self.cross_encoder = None
        
        # Skip loading if CrossEncoder class is unavailable or model name is empty
        if not CrossEncoder or not self.cross_encoder_name:
            logger.info("Cross-Encoder reranking disabled (lightweight mode or no model configured).")
            return
            
        try:
            logger.info(f"Loading Cross-Encoder model: {self.cross_encoder_name}...")
            try:
                self.cross_encoder = CrossEncoder(self.cross_encoder_name, local_files_only=True)
                logger.info("Cross-Encoder model loaded successfully from local cache.")
            except Exception as e:
                logger.warning(f"Failed to load Cross-Encoder from local cache ({e}). Attempting online download/update...")
                self.cross_encoder = CrossEncoder(self.cross_encoder_name, local_files_only=False)
                logger.info("Cross-Encoder model loaded successfully online.")
        except Exception as e:
            logger.warning(f"Failed to load Cross-Encoder: {e}. Falling back to standard keyword-boosted cosine search.")

    def _compute_keyword_boost(self, query: str, text: str) -> float:
        """
        Computes a keyword frequency matching boost.
        Identifies key terms in query and counts their occurrences in chunk text.
        """
        if not query or not text:
            return 0.0
            
        # Clean query, split into words, remove common short words/stop words
        stop_words = {"what", "is", "the", "a", "an", "of", "in", "to", "for", "with", "on", "at", "by", "from", "and", "or", "but", "this", "that"}
        words = re.findall(r'\w+', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        
        if not keywords:
            return 0.0
            
        text_lower = text.lower()
        match_count = 0
        for kw in keywords:
            # Count occurrences of keyword in text
            match_count += len(re.findall(re.escape(kw), text_lower))
            
        # Add score boost: 0.01 per keyword hit, capped at 0.15
        boost = min(match_count * 0.01, 0.15)
        return boost

    def retrieve(
        self, 
        query: str, 
        top_k: int = 3, 
        fetch_k: int = 8
    ) -> List[Dict]:
        """
        Performs hybrid retrieval:
        1. Fetch top fetch_k candidates from FAISS vector search.
        2. Apply term matching keyword boost.
        3. Rerank using Cross-Encoder.
        4. Return top_k results.
        """
        # Fetch candidate chunks from vector DB
        candidates = self.vector_db.search(query, top_k=fetch_k)
        if not candidates:
            return []

        # Apply keyword and profile-routing boosts to candidates
        query_lower = query.lower()
        profile_terms = {
            "president", "representative", "director", "officer", "officers", 
            "founder", "founded", "executive", "board", "leadership", "sekiguchi", "関口",
            "located", "location", "address", "headquarters", "head office", "established", 
            "establishment", "capital", "employee", "employees", "staff", "outline"
        }
        is_profile_query = any(term in query_lower for term in profile_terms)
        
        for cand in candidates:
            boost = self._compute_keyword_boost(query, cand["text_en"])
            
            # Massive boost for company outline/profile gold chunks on profile metadata queries
            if is_profile_query:
                cand_text = cand["text_en"].lower()
                is_outline_source = "outline" in cand["url"].lower() or "outline" in cand["title"].lower() or "local-json-cache" in cand["url"].lower()
                is_outline_content = "greetings from the president" in cand_text or "company outline" in cand_text or "representative director" in cand_text
                if is_outline_source and is_outline_content:
                    boost += 0.35
                    
            cand["base_score"] = cand["score"]
            cand["score"] += boost
            
        # Rerank candidates using Cross-Encoder if available
        if self.cross_encoder:
            try:
                # Prepare input pairs for cross encoder: (query, document)
                pairs = [[query, cand["text_en"]] for cand in candidates]
                
                # Predict relevance scores (higher is more relevant)
                rerank_scores = self.cross_encoder.predict(pairs)
                
                # Apply rerank scores
                for idx, score in enumerate(rerank_scores):
                    # Sigmoid normalization (CrossEncoder MS-Marco can output unbounded values, we map them loosely)
                    # or keep it as direct ranking key.
                    candidates[idx]["score"] = float(score)
                    
                # Sort candidates by new Cross-Encoder score (descending)
                candidates.sort(key=lambda x: x["score"], reverse=True)
                logger.info("Successfully reranked documents using Cross-Encoder.")
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Using keyword-boosted cosine scores.")
                candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            # Sort by keyword-boosted score
            candidates.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k
        return candidates[:top_k]
