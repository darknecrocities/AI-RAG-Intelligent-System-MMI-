import re
from typing import List, Dict

def count_approx_tokens(text: str) -> int:
    """
    Approximates token count.
    For English, words are split by whitespace (1 word ~ 1.3 tokens).
    For CJK characters, each character is counted as a token.
    """
    if not text:
        return 0
        
    # Count CJK (Chinese, Japanese, Korean) characters
    cjk_count = len(re.findall(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\u3400-\u4dbf\uff00-\uffef]', text))
    
    # Remove CJK characters to count English/other words
    non_cjk_text = re.sub(r'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\u3400-\u4dbf\uff00-\uffef]', ' ', text)
    words = non_cjk_text.split()
    word_count = len(words)
    
    # Total approximate tokens
    approx_tokens = int(word_count * 1.3) + cjk_count
    return max(approx_tokens, 1)

def semantic_chunk_document(
    document: dict, 
    chunk_size: int = 600, 
    chunk_overlap: int = 100
) -> List[Dict]:
    """
    Splits a document (with title, content, headings, url, language, translated flag)
    into semantically chunked blocks. Preserves titles, headings, and source URL context.
    """
    url = document.get("url", "")
    title = document.get("title", "")
    content_en = document.get("content_en", "") # English version for embeddings/search
    content_ja = document.get("content_ja", "") # Japanese version for display
    headings = document.get("headings", [])
    
    if not content_en:
        content_en = document.get("content", "")
    if not content_ja:
        content_ja = document.get("content", "")

    # We split the English content into paragraphs
    paragraphs_en = content_en.split("\n\n")
    # Align Japanese paragraphs if possible, otherwise use split paragraphs as well
    paragraphs_ja = content_ja.split("\n\n")
    
    # If paragraph count mismatch, we'll fall back to just splitting English, and keeping full Japanese context or mapping
    has_aligned = len(paragraphs_en) == len(paragraphs_ja)
    
    chunks = []
    current_paragraphs_en = []
    current_paragraphs_ja = []
    current_tokens = 0
    
    active_section = "General"
    
    # Helper to check if a paragraph is a heading
    def clean_heading(text: str) -> str:
        return re.sub(r'^[#\s\-*]+', '', text).strip()

    for idx, p_en in enumerate(paragraphs_en):
        p_ja = paragraphs_ja[idx] if (has_aligned and idx < len(paragraphs_ja)) else ""
        
        # Check if this paragraph is a heading
        is_heading = False
        p_en_clean = clean_heading(p_en)
        for h in headings:
            h_text_clean = clean_heading(h.get("text", ""))
            if p_en_clean.lower() == h_text_clean.lower() or h_text_clean.lower() in p_en_clean.lower():
                is_heading = True
                active_section = h_text_clean
                break
        
        p_tokens = count_approx_tokens(p_en)
        
        # If adding this paragraph exceeds the chunk size, save current chunk
        if current_tokens + p_tokens > chunk_size and current_paragraphs_en:
            # Join paragraphs to create chunk texts
            chunk_text_en = "\n\n".join(current_paragraphs_en)
            chunk_text_ja = "\n\n".join(current_paragraphs_ja) if current_paragraphs_ja else chunk_text_en
            
            # Format context text block to be embedded/searched
            # Prepending title, section, and URL so that the vector space contains these references
            searchable_text = (
                f"Source: {url}\n"
                f"Title: {title}\n"
                f"Section: {active_section}\n\n"
                f"{chunk_text_en}"
            )
            
            chunks.append({
                "url": url,
                "title": title,
                "section": active_section,
                "text_en": searchable_text, # text containing meta-references for search
                "text_ja": chunk_text_ja,   # matching original Japanese text
                "content_en_raw": chunk_text_en,
                "tokens": count_approx_tokens(searchable_text)
            })
            
            # Keep overlap paragraphs
            overlap_p_en = []
            overlap_p_ja = []
            overlap_tokens = 0
            # Walk backwards and grab paragraphs up to overlap limit
            for o_idx in range(len(current_paragraphs_en) - 1, -1, -1):
                o_text = current_paragraphs_en[o_idx]
                o_tokens = count_approx_tokens(o_text)
                if overlap_tokens + o_tokens > chunk_overlap:
                    break
                overlap_p_en.insert(0, o_text)
                if current_paragraphs_ja and o_idx < len(current_paragraphs_ja):
                    overlap_p_ja.insert(0, current_paragraphs_ja[o_idx])
                overlap_tokens += o_tokens
                
            current_paragraphs_en = overlap_p_en
            current_paragraphs_ja = overlap_p_ja
            current_tokens = overlap_tokens

        current_paragraphs_en.append(p_en)
        if p_ja:
            current_paragraphs_ja.append(p_ja)
        current_tokens += p_tokens

    # Add final chunk
    if current_paragraphs_en:
        chunk_text_en = "\n\n".join(current_paragraphs_en)
        chunk_text_ja = "\n\n".join(current_paragraphs_ja) if current_paragraphs_ja else chunk_text_en
        searchable_text = (
            f"Source: {url}\n"
            f"Title: {title}\n"
            f"Section: {active_section}\n\n"
            f"{chunk_text_en}"
        )
        chunks.append({
            "url": url,
            "title": title,
            "section": active_section,
            "text_en": searchable_text,
            "text_ja": chunk_text_ja,
            "content_en_raw": chunk_text_en,
            "tokens": count_approx_tokens(searchable_text)
        })
        
    return chunks
