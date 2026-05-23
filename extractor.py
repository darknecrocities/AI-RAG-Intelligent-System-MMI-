import logging
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger(__name__)

def clean_html_with_bs4(html_content: str) -> dict:
    """
    Fallback method using BeautifulSoup to extract main text, titles, and headers.
    It explicitly removes common boilerplate elements.
    """
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script, style, header, footer, nav, and noscript elements
        for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            element.decompose()
            
        # Target specific boilerplate elements by ID/Class commonly found in MMI site
        for boiler_id in ["masthead", "colophon", "noscript"]:
            el = soup.find(id=boiler_id)
            if el:
                el.decompose()
                
        for boiler_class in ["site-header", "site-footer", "nav-button-wrap", "nav-button", "navi", "slider-nav", "slider-container"]:
            for el in soup.find_all(class_=boiler_class):
                el.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.h1:
            title = soup.h1.get_text().strip()
            
        # Extract headings (H1, H2, H3)
        headings = []
        for tag in ["h1", "h2", "h3"]:
            for h in soup.find_all(tag):
                text = h.get_text().strip()
                if text:
                    headings.append({"level": tag, "text": text})

        # Get main text
        paragraphs = []
        # Look for paragraphs or divs with text
        for p in soup.find_all(["p", "div", "section", "article"]):
            # Avoid nesting text duplicate issues by fetching paragraph/direct text blocks
            if p.name == "p":
                text = p.get_text().strip()
                if text:
                    paragraphs.append(text)
            elif p.name in ["div", "section", "article"]:
                # Only take text if there are no child block elements to prevent double reading
                if not p.find(["p", "div", "section"]):
                    text = p.get_text().strip()
                    if text:
                        paragraphs.append(text)

        content = "\n\n".join(paragraphs)
        
        return {
            "title": title,
            "content": content,
            "headings": headings
        }
    except Exception as e:
        logger.error(f"Error in BeautifulSoup fallback extraction: {e}")
        return {"title": "", "content": "", "headings": []}

def extract_content(html_content: str, url: str) -> dict:
    """
    Main extraction interface. Attempts trafilatura first for high-quality clean extraction,
    and falls back to BeautifulSoup if results are empty.
    """
    result = {
        "url": url,
        "title": "",
        "content": "",
        "headings": [],
        "extraction_method": "trafilatura"
    }
    
    if not html_content:
        return result

    try:
        # Trafilatura extraction
        extracted_text = trafilatura.extract(
            html_content,
            include_comments=False,
            include_tables=True,
            no_fallback=True
        )
        
        # Trafilatura metadata extraction
        metadata = trafilatura.extract_metadata(html_content)
        title = metadata.title if metadata and metadata.title else ""
        
        if extracted_text and len(extracted_text.strip()) > 50:
            result["title"] = title
            result["content"] = extracted_text
            
            # Extract headings via a quick BeautifulSoup pass on headings only
            soup = BeautifulSoup(html_content, "html.parser")
            headings = []
            for tag in ["h1", "h2", "h3"]:
                for h in soup.find_all(tag):
                    text = h.get_text().strip()
                    if text:
                        headings.append({"level": tag, "text": text})
            result["headings"] = headings
            return result
    except Exception as e:
        logger.warning(f"Trafilatura failed or returned empty for {url}: {e}. Falling back to BeautifulSoup.")

    # Fallback to BeautifulSoup
    bs_result = clean_html_with_bs4(html_content)
    result["title"] = bs_result["title"]
    result["content"] = bs_result["content"]
    result["headings"] = bs_result["headings"]
    result["extraction_method"] = "beautifulsoup"
    
    return result
