import json
import logging
from typing import List, Dict, Generator
from ollama import Client
import config

logger = logging.getLogger(__name__)

class OllamaEngine:
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.client = Client(host=config.OLLAMA_API_URL)
        self._check_ollama_connection()

    def _check_ollama_connection(self) -> bool:
        """
        Verifies if Ollama is running and handles model status logging.
        """
        try:
            self.client.list()
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
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1
                }
            )
            rewritten = response.response.strip()
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

    def build_prompt(self, context: str, query: str, history: List[Dict[str, str]] = None) -> List[Dict]:
        """
        Constructs the structured message list for Ollama Chat API.
        """
        messages = []
        
        # System instructions
        system_content = (
            "You are a helpful, enthusiastic, and highly knowledgeable enterprise assistant for MMI (株式会社マン・マシンインターフェース, Man Machine Interface).\n"
            "INSTRUCTIONS:\n"
            "1. If the question is conversational (e.g., greetings, small talk) or refers to your previous messages, reply naturally, warmly, and briefly based on the CHAT HISTORY.\n"
            "2. If the question is about the company, its products, services, or philosophy, answer using ONLY the provided CONTEXT. "
            "If the information is not present in the CONTEXT, politely explain that you don't have that specific information right now, but offer to help with other related topics. "
            "Always maintain a polite, professional, and friendly tone."
        )
        messages.append({"role": "system", "content": system_content})
        
        # Append chat history before the current message
        if history:
            for msg in history:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
                
        # Append the current query and context
        user_content = ""
        if context.strip():
            user_content += f"CONTEXT:\n{context}\n\n"
        user_content += f"QUESTION: {query}"
        messages.append({"role": "user", "content": user_content})
        
        return messages

    def generate(self, messages: List[Dict]) -> Dict:
        """
        Generates a response synchronously using /api/chat.
        """
        # Backward compatibility check
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.2
                }
            )
            return {
                "text": response.message.content,
                "done": True
            }
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return {"text": f"Error contacting local LLM daemon: {e}", "done": True}

    def generate_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """
        Streams response tokens back in Real-time using /api/chat.
        """
        # Backward compatibility check
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        try:
            response_stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.2
                }
            )
            for chunk in response_stream:
                token = chunk.message.content
                yield token
        except Exception as e:
            logger.error(f"Streaming error from Ollama: {e}")
            yield f"\n[Error communicating with Ollama: {e}]"

