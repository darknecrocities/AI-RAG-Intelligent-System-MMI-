import os
import sys
import time
import logging
from typing import List, Dict

# Set logging to a file to keep terminal clean
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    filename="chatbot_cli.log",
    filemode="w",
    encoding="utf-8"
)

# Print an ASCII banner
BANNER = """
=========================================================
          MMI-KNOWLEDGE RAG CLI CHATBOT
=========================================================
"""

def check_ollama():
    import requests
    import config
    try:
        r = requests.get(config.OLLAMA_API_URL, timeout=3)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    return False

def run_full_scraping():
    print("\n[+] Starting full web scraping & knowledge ingestion...")
    print("    This will crawl https://www.mmi-sc.co.jp/ recursively,")
    print("    extract and translate content, and index it into FAISS.")
    print("    Please wait, this may take 1-3 minutes depending on your network...")
    
    try:
        from crawler import WebCrawler
        from extractor import extract_content
        from translator import TranslationEngine
        from chunker import semantic_chunk_document
        from vector_db import VectorDB
        import config
        
        # Load components
        crawler = WebCrawler()
        translator = TranslationEngine()
        db = VectorDB()
        
        print("\n[1/3] Crawling all pages on the domain (discovering links)...")
        # Crawl all pages
        crawl_results = crawler.crawl(force_recrawl=True)
        print(f"      Successfully retrieved {len(crawl_results)} pages.")
        
        print("\n[2/3] Extracting, translating, and chunking contents...")
        chunks_to_embed = []
        
        # We clear the index first to build a clean index
        db.clear()
        
        for idx, (url, page) in enumerate(crawl_results.items()):
            print(f"      [{idx+1}/{len(crawl_results)}] Processing URL: {url} ...")
            
            # Extract content
            extracted = extract_content(page["html"], url)
            original_text = extracted["content"]
            title = extracted["title"]
            
            if not original_text.strip():
                continue
                
            # Translate if Japanese
            if translator.is_japanese(original_text):
                translated_text = translator.translate_text(original_text)
                translated_title = translator.translate_text(title)
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
            
            # Semantic chunking
            chunks = semantic_chunk_document(
                document=doc,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            chunks_to_embed.extend(chunks)
            
        print(f"\n[3/3] Generating embeddings and building FAISS Vector store for {len(chunks_to_embed)} chunks...")
        if chunks_to_embed:
            db.add_chunks(chunks_to_embed)
            print(f"      Success! Vector index saved to '{config.VECTOR_DB_PATH}'.")
        else:
            print("      Warning: No text chunks generated.")
            
        print("\n[✓] Ingestion Completed Successfully!")
    except Exception as e:
        print(f"\n[✗] Ingestion failed: {e}")
        import traceback
        traceback.print_exc()

def interactive_chat():
    from vector_db import VectorDB
    from retriever import HybridRetriever
    from llm import OllamaEngine
    
    print("\n[+] Loading local models and FAISS index...")
    db = VectorDB()
    retriever = HybridRetriever(db)
    llm = OllamaEngine()
    chat_history = []
    
    if db.index is None or db.index.ntotal == 0:
        print("\n[!] WARNING: The vector database is empty.")
        print("    Please run the web scraper first (Option 1) to build the database.")
    
    print("\n[+] Ready! Enter 'exit' or 'quit' to stop.")
    print("---------------------------------------------------------")
    
    while True:
        try:
            query = input("\nUser > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            print("\nAssistant > ", end="")
            
            # Smart Query Rewriting
            optimized_query = llm.rewrite_query(query)
            
            # Retrieval
            retrieved_chunks = retriever.retrieve(optimized_query, top_k=4)
            if not retrieved_chunks:
                print("Not found in knowledge base.")
                continue
                
            # Context compression & prompt preparation
            compressed_context = llm.compress_context(retrieved_chunks)
            prompt = llm.build_prompt(compressed_context, query, chat_history)
            
            # Streaming answer token-by-token
            collected_response = ""
            for token in llm.generate_stream(prompt):
                sys.stdout.write(token)
                sys.stdout.flush()
                collected_response += token
                
            # Update history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": collected_response})
            chat_history = chat_history[-8:]
                
            # Format and display sources/citations neatly
            print("\n\nSources / Citations:")
            seen_urls = set()
            count = 1
            for chunk in retrieved_chunks:
                url = chunk["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    match_percent = Math_percent = int(chunk["score"] * 100)
                    section = chunk["section"] or "General"
                    title = chunk["title"] or "MMI Page"
                    print(f"  [{count}] {title} > {section}")
                    print(f"      URL: {url} ({match_percent}% match)")
                    count += 1
            print("---------------------------------------------------------")
            
        except EOFError:
            print("\nInput stream ended. Exiting.")
            break
        except KeyboardInterrupt:
            print("\nSession interrupted. Exit.")
            break
        except Exception as e:
            print(f"\n[Error occurred]: {e}")

def main():
    print(BANNER)
    
    # Check Ollama
    print("Checking local Ollama connection...")
    if check_ollama():
        print("✓ Ollama status: Running.")
    else:
        print("✗ Ollama status: Not running!")
        print("  Please make sure Ollama is launched and running locally before starting.")
        sys.exit(1)
        
    print("\nSelect an option:")
    print("  1. Scrape & Index the entire MMI website (Build Database)")
    print("  2. Chat with RAG (Use existing index)")
    print("  3. Exit")
    
    try:
        choice = input("\nEnter choice [1-3]: ").strip()
        if choice == "1":
            run_scraping = True
            run_full_scraping()
            interactive_chat()
        elif choice == "2":
            interactive_chat()
        else:
            print("Exit.")
    except KeyboardInterrupt:
        print("\nExit.")

if __name__ == "__main__":
    main()
