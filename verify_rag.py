import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAG-Verification")

def run_tests():
    logger.info("=== STARTING MMI RAG SYSTEM COMPONENT TESTS ===")
    
    # 1. Test Config Loading
    try:
        import config
        logger.info(f"✓ Config loaded. Target base URL: {config.BASE_URL}")
        logger.info(f"✓ Data folder: {config.DATA_DIR}")
    except Exception as e:
        logger.error(f"✗ Config load failed: {e}")
        return False
        
    # 2. Test Content Extraction Engine
    try:
        import extractor
        mock_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Page Title</title></head>
        <body>
          <header id="masthead"><h2>Navigation Header (Boilerplate)</h2></header>
          <div class="content">
            <h1>Welcome to MMI</h1>
            <p>This is a paragraph containing information about embedded systems development.</p>
            <h2>EE Division</h2>
            <p>Our Electrical Engineering division designs custom circuit boards.</p>
          </div>
          <footer id="colophon"><p>Footer Copyright (Boilerplate)</p></footer>
        </body>
        </html>
        """
        extracted = extractor.extract_content(mock_html, "https://www.mmi-sc.co.jp/test-page")
        assert "Welcome to MMI" in extracted["content"], "Content missing expected text"
        assert "EE Division" in [h["text"] for h in extracted["headings"]], "Headings extraction failed"
        assert "Navigation Header" not in extracted["content"], "Boilerplate header was not stripped"
        assert "Footer Copyright" not in extracted["content"], "Boilerplate footer was not stripped"
        
        logger.info("✓ Content Extraction: Successful (boilerplates stripped, text extracted).")
        logger.info(f"  Method used: {extracted['extraction_method']}, Title: '{extracted['title']}'")
    except Exception as e:
        logger.error(f"✗ Content Extraction failed: {e}")
        return False

    # 3. Test Language Detection & Translation
    try:
        import translator
        engine = translator.TranslationEngine()
        
        ja_text = "マン・マシンインターフェースは組込みソフトウェアの開発を行っています。"
        en_text = "This is a clean English sentence."
        
        # Test detection
        assert engine.is_japanese(ja_text) == True, "Failed to detect Japanese"
        assert engine.is_japanese(en_text) == False, "Incorrectly flagged English as Japanese"
        logger.info("✓ Language Detection: Successful.")
        
        # Test translation (free endpoint online, if connection is available)
        logger.info("Testing Japanese-to-English translation online...")
        translated = engine.translate_text(ja_text)
        logger.info(f"  Original: {ja_text}")
        logger.info(f"  Translated: {translated}")
        assert len(translated) > 0 and translated != ja_text, "Translation returned empty or unmodified text"
        logger.info("✓ Translation Layer: Successful.")
    except Exception as e:
        logger.error(f"✗ Language & Translation test failed: {e}")
        logger.info("  Note: Online translation might fail if disconnected from internet or rate-limited.")
        return False

    # 4. Test Chunker
    try:
        import chunker
        mock_doc = {
            "url": "https://www.mmi-sc.co.jp/test",
            "title": "Test Page",
            "content_ja": "マン・マシン\n\nインターフェースの開発",
            "content_en": "Man-Machine\n\nInterface development",
            "headings": [{"level": "h2", "text": "Philosophy"}]
        }
        chunks = chunker.semantic_chunk_document(mock_doc, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 0, "Chunker returned empty chunk list"
        logger.info(f"✓ Chunker: Successful. Generated {len(chunks)} chunks.")
        logger.info(f"  Chunk 0 URL: {chunks[0]['url']}, Section: {chunks[0]['section']}")
    except Exception as e:
        logger.error(f"✗ Chunker test failed: {e}")
        return False

    # 5. Test Embeddings and Vector DB
    try:
        import config
        config.VECTOR_DB_PATH = "./data/test_faiss_index"
        import vector_db
        import chunker
        
        logger.info("Initializing FAISS Vector Store...")
        db = vector_db.VectorDB()
        
        # Add test chunks
        test_chunks = [
            {
                "url": "https://www.mmi-sc.co.jp/aws",
                "title": "AWS Partnership",
                "section": "AWS Integration",
                "text_en": "Source: https://www.mmi-sc.co.jp/aws\nTitle: AWS Partnership\nSection: AWS Integration\n\nMMI is an AWS Partner providing cloud DICOM gateway software.",
                "text_ja": "MMIはAWSパートナーで、DICOMゲートウェイを提供しています。",
                "content_en_raw": "MMI is an AWS Partner providing cloud DICOM gateway software."
            },
            {
                "url": "https://www.mmi-sc.co.jp/about",
                "title": "Company Outline",
                "section": "Outline",
                "text_en": "Source: https://www.mmi-sc.co.jp/about\nTitle: Company Outline\nSection: Outline\n\nMMI was founded in 1985 and is headquartered in Tokyo, Japan.",
                "text_ja": "MMIは1985年に設立され、東京に本社があります。",
                "content_en_raw": "MMI was founded in 1985 and is headquartered in Tokyo, Japan."
            }
        ]
        
        db.clear()
        db.add_chunks(test_chunks)
        
        # Test Search
        query = "cloud DICOM gateway"
        results = db.search(query, top_k=2)
        assert len(results) > 0, "Vector search returned no results"
        assert "AWS" in results[0]["text_en"], "Vector search ranked wrong page on top"
        logger.info("✓ Vector DB: Successful. Embedding generation, FAISS flat index build, and cosine search completed.")
        logger.info(f"  Top Match URL: {results[0]['url']}, Score: {results[0]['score']:.4f}")
    except Exception as e:
        logger.error(f"✗ Vector DB test failed: {e}")
        return False

    # 6. Test Reranking retriever
    try:
        import retriever
        logger.info("Initializing Hybrid Reranking Retriever...")
        r = retriever.HybridRetriever(db)
        
        res = r.retrieve("Tokyo headquarters 1985")
        assert len(res) > 0, "Retriever failed"
        assert "Tokyo" in res[0]["text_en"], "Reranking ranked wrong page on top"
        logger.info("✓ Hybrid Reranking Retriever: Successful.")
    except Exception as e:
        logger.error(f"✗ Hybrid Retriever test failed: {e}")
        return False

    # 7. Check HuggingFace Engine status
    try:
        import llm
        engine = llm.HuggingFaceEngine()
        hf_ok = engine._check_hf_connection()
        if hf_ok:
            logger.info("✓ LLM Engine: Connection to Hugging Face Space verified.")
        else:
            logger.warning("! LLM Engine: Hugging Face Space connection could not be established.")
    except Exception as e:
        logger.error(f"✗ LLM Engine test failed: {e}")
        return False

    # Clean up test database directory
    try:
        import os
        import shutil
        if os.path.exists("./data/test_faiss_index"):
            shutil.rmtree("./data/test_faiss_index")
            logger.info("✓ Cleanup: Removed temporary test FAISS directory.")
    except Exception as e:
        logger.warning(f"Failed to remove test FAISS directory: {e}")

    logger.info("=== ALL CODE COMPONENT TESTS PASSED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
