import json
import logging
from typing import List, Dict, Generator
from gradio_client import Client
import config

logger = logging.getLogger(__name__)

class HuggingFaceEngine:
    def __init__(self):
        self.hf_url = config.HF_SPACE_URL
        self.client = Client(self.hf_url)
        self._check_hf_connection()

    def _check_hf_connection(self) -> bool:
        """
        Verifies if the Hugging Face Space is reachable.
        """
        try:
            # A simple quick call or just relies on Client init which fetches API info
            logger.info(f"Successfully connected to Hugging Face Space at {self.hf_url}.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Hugging Face Space at {self.hf_url}: {e}")
            return False

    def rewrite_query(self, query: str) -> str:
        """
        Smart Query Rewriting.
        Uses the HF Space to optimize the query.
        """
        prompt = (
            "You are a search query optimizer.\n"
            "Your task is to take a user question, expand keywords, fix spelling/grammar, and translate it to English if it is in Japanese.\n"
            "Respond with ONLY the optimized English search query. Do not add any conversational text, quotes, or preambles.\n\n"
            f"Original Question: {query}\n"
            "Optimized English Query:"
        )
        
        try:
            logger.info("Rewriting user query via HF Space...")
            response = self.client.predict(
                message=prompt,
                api_name="/predict"
            )
            rewritten = response.strip()
            if rewritten:
                rewritten = rewritten.strip('"').strip("'")
                logger.info(f"Rewritten query: '{query}' -> '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Failed to rewrite query: {e}. Using original query.")
            
        return query


    def compress_context(self, chunks: List[Dict], max_tokens: int = 2500) -> str:
        """
        Compresses retrieved context.
        """
        context_blocks = []
        accumulated_tokens = 0
        
        for chunk in chunks:
            text = chunk["text_en"]
            approx_tokens = len(text) // 4
            
            if accumulated_tokens + approx_tokens > max_tokens:
                logger.info("Context compressed: hit token limit.")
                break
                
            context_blocks.append(text)
            accumulated_tokens += approx_tokens
            
        return "\n\n---\n\n".join(context_blocks)

    def build_prompt(self, context: str, query: str, history: List[Dict[str, str]] = None) -> Dict:
        """
        Constructs the structured message for the Gradio /predict endpoint.
        The HF Space only accepts a single 'message' parameter, so we pack
        everything (system instructions, context, history, and query) into one string.
        """
        system_prefix = (
            "You are a helpful, enthusiastic, and highly knowledgeable enterprise assistant for MMI (株式会社マン・マシンインターフェース, Man Machine Interface).\n"
            "INSTRUCTIONS:\n"
            "1. If the question is conversational, reply naturally, warmly, and briefly based on the CHAT HISTORY.\n"
            "2. If the question is about the company, its products, services, or philosophy, answer using ONLY the provided CONTEXT. "
            "If the information is not present in the CONTEXT, politely explain that you don't have that specific information right now.\n\n"
        )
        
        user_content = system_prefix

        # Include chat history as part of the prompt text
        if history:
            user_content += "CHAT HISTORY:\n"
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                user_content += f"  {role}: {content}\n"
            user_content += "\n"

        if context.strip():
            user_content += f"CONTEXT:\n{context}\n\n"
        user_content += f"QUESTION: {query}"
        
        return {
            "message": user_content
        }

    def generate(self, prompt_data: Dict) -> Dict:
        """
        Generates a response synchronously using HF Gradio Client.
        """
        try:
            response = self.client.predict(
                message=prompt_data["message"],
                api_name="/predict"
            )
            return {
                "text": response,
                "done": True
            }
        except Exception as e:
            logger.error(f"Error calling HF Space: {e}")
            return {"text": f"Error contacting Hugging Face Space: {e}", "done": True}

    def generate_stream(self, prompt_data: Dict) -> Generator[str, None, None]:
        """
        Streams response tokens back in Real-time.
        Since /predict is a non-streaming endpoint, we call it synchronously
        and yield the full response as a single chunk.
        """
        try:
            response = self.client.predict(
                message=prompt_data["message"],
                api_name="/predict"
            )
            yield response
                        
        except Exception as e:
            logger.error(f"Streaming error from HF Space: {e}")
            yield f"\n[Error communicating with Hugging Face Space: {e}]"
