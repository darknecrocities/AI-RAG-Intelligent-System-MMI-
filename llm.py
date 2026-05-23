import json
import logging
from typing import List, Dict, Generator
import requests
import config

logger = logging.getLogger(__name__)

class OllamaEngine:
    def __init__(self):
        self.api_url = f"{config.OLLAMA_API_URL}/api/generate"
        self.model = config.OLLAMA_MODEL
        self._check_ollama_connection()

    def _check_ollama_connection(self) -> bool:
        """
        Verifies if Ollama is running and handles model status logging.
        """
        try:
            r = requests.get(config.OLLAMA_API_URL, timeout=3)
            if r.status_code == 200:
                logger.info("Successfully connected to local Ollama daemon.")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to local Ollama daemon at {config.OLLAMA_API_URL}: {e}")
        return False

    def rewrite_query(self, query: str) -> str:
        """
        Smart Query Rewriting:
        Expands user keywords, corrects grammar, and translates to English if input in Japanese.
        """
        prompt = (
            "You are a search query optimizer.\n"
            "Your task is to take a user question, expand keywords, fix spelling/grammar, and translate it to English if it is in Japanese.\n"
            "Respond with ONLY the optimized English search query. Do not add any conversational text, quotes, or preambles.\n\n"
            f"Original Question: {query}\n"
            "Optimized English Query:"
        )
        
        try:
            logger.info("Rewriting user query...")
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            }
            r = requests.post(self.api_url, json=payload, timeout=10)
            if r.status_code == 200:
                rewritten = r.json().get("response", "").strip()
                if rewritten:
                    # Strip any outer quotes if LLM added them
                    rewritten = rewritten.strip('"').strip("'")
                    logger.info(f"Rewritten query: '{query}' -> '{rewritten}'")
                    return rewritten
        except Exception as e:
            logger.warning(f"Failed to rewrite query: {e}. Using original query.")
            
        return query

    def compress_context(self, chunks: List[Dict], max_tokens: int = 2500) -> str:
        """
        Compresses retrieved context to fit model token boundaries.
        Merges chunks into a unified string. If the content is too large, 
        takes only the top chunks until the token budget is filled.
        """
        context_blocks = []
        accumulated_tokens = 0
        
        # Approximate tokens in each chunk and aggregate
        for chunk in chunks:
            text = chunk["text_en"]
            # Estimate token count (chars / 4)
            approx_tokens = len(text) // 4
            
            if accumulated_tokens + approx_tokens > max_tokens:
                logger.info("Context compressed: hit token limit.")
                break
                
            context_blocks.append(text)
            accumulated_tokens += approx_tokens
            
        return "\n\n---\n\n".join(context_blocks)

    def build_prompt(self, context: str, query: str, history: List[Dict[str, str]] = None) -> str:
        """
        Constructs the standard RAG prompt template.
        """
        history_str = ""
        if history:
            for msg in history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                history_str += f"{role}: {content}\n"
        else:
            history_str = "No previous history."

        prompt = (
            "You are a precise enterprise knowledge assistant for MMI (株式会社マン・マシンインターフェース).\n"
            "INSTRUCTIONS:\n"
            "1. If the question is conversational (e.g., greetings, talking about past chat, small talk) or refers to your previous messages, reply naturally, warmly, and briefly based on the CHAT HISTORY.\n"
            "2. If the question is about the company, its products, or services, answer using ONLY the provided CONTEXT. If not present in the CONTEXT, state 'Not found in knowledge base'. Be concise and factual.\n\n"
            f"CHAT HISTORY:\n{history_str}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            "Answer:"
        )
        return prompt

    def generate(self, prompt: str) -> Dict:
        """
        Generates a response synchronously.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        
        try:
            r = requests.post(self.api_url, json=payload, timeout=30)
            if r.status_code == 200:
                response_json = r.json()
                return {
                    "text": response_json.get("response", ""),
                    "done": True
                }
            else:
                return {"text": f"Error from Ollama: {r.status_code}", "done": True}
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return {"text": f"Error contacting local LLM daemon: {e}", "done": True}

    def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Streams response tokens back in Real-time.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2
            }
        }
        
        try:
            r = requests.post(self.api_url, json=payload, stream=True, timeout=30)
            if r.status_code == 200:
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        yield token
                        if chunk.get("done", False):
                            break
            else:
                yield f"Error from Ollama: {r.status_code}"
        except Exception as e:
            logger.error(f"Streaming error from Ollama: {e}")
            yield f"\n[Error communicating with Ollama: {e}]"
