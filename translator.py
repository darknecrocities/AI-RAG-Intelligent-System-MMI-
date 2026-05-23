import os
import json
import time
import logging
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import config

# Set seed for langdetect for reproducible results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class TranslationEngine:
    def __init__(self):
        self.cache_path = config.TRANSLATION_CACHE_PATH
        self.cache = self._load_cache()
        
    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load translation cache: {e}")
        return {}

    def _save_cache(self):
        try:
            # Ensure folder exists
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save translation cache: {e}")

    def is_japanese(self, text: str) -> bool:
        """
        Detects if text contains Japanese character sequences or if langdetect flags it as ja.
        """
        if not text or len(text.strip()) == 0:
            return False
            
        # Fast character check for Japanese (Hiragana, Katakana, Kanji)
        # Hiragana: 3040-309F, Katakana: 30A0-30FF, CJK Unified Ideographs (Kanji): 4E00-9FBF
        has_japanese_chars = any(
            (0x3040 <= ord(char) <= 0x309F) or 
            (0x30A0 <= ord(char) <= 0x30FF) or 
            (0x4E00 <= ord(char) <= 0x9FFF)
            for char in text[:1000] # Check first 1000 chars
        )
        if has_japanese_chars:
            return True

        try:
            lang = detect(text)
            return lang == "ja"
        except Exception:
            return False

    def translate_chunk(self, text: str, max_retries: int = 3) -> str:
        """
        Translates a single text string up to 4500 characters using GoogleTranslator.
        Includes exponential backoff retries.
        """
        if not text or len(text.strip()) == 0:
            return ""

        # Check in-memory/file cache
        if text in self.cache:
            return self.cache[text]

        translator = GoogleTranslator(source="ja", target="en")
        
        delay = 1.0
        for attempt in range(max_retries):
            try:
                # Add a small delay between requests to prevent Google Translate rate-limiting
                time.sleep(0.6)
                translated = translator.translate(text)
                if translated:
                    # Update cache
                    self.cache[text] = translated
                    self._save_cache()
                    return translated
            except Exception as e:
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Translation failed after {max_retries} attempts.")
                    
        return text # Fallback to original text on failure

    def translate_text(self, text: str) -> str:
        """
        Translates long text blocks by splitting them into safe chunks (under 4000 characters),
        translating each chunk, and combining the results.
        """
        if not text or len(text.strip()) == 0:
            return ""
            
        if not self.is_japanese(text):
            return text

        # Split text into paragraphs first to preserve boundaries
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0
        
        for p in paragraphs:
            # If paragraph itself is too large (rare, but possible), split by sentences
            if len(p) > 4000:
                sentences = p.replace("。", "。\n").replace(". ", ".\n").split("\n")
                for s in sentences:
                    if current_len + len(s) > 4000:
                        if current_chunk:
                            chunks.append("\n".join(current_chunk))
                        current_chunk = [s]
                        current_len = len(s)
                    else:
                        current_chunk.append(s)
                        current_len += len(s)
            else:
                if current_len + len(p) > 4000:
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    current_chunk = [p]
                    current_len = len(p)
                else:
                    current_chunk.append(p)
                    current_len += len(p) + 2 # account for \n\n

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        # Translate each chunk and assemble
        translated_chunks = []
        for chunk in chunks:
            translated_chunks.append(self.translate_chunk(chunk))
            
        return "\n\n".join(translated_chunks)
