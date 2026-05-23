import os
import re
import json
import hashlib
import logging
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import config

logger = logging.getLogger(__name__)

class WebCrawler:
    def __init__(self):
        self.base_url = config.BASE_URL
        self.domain = urlparse(self.base_url).netloc
        self.cache_path = config.CRAWL_CACHE_PATH
        self.session = requests.Session()
        # Set user-agent to look like a friendly search bot
        self.session.headers.update({
            "User-Agent": "MMI-Knowledge-RAG-Crawler/1.0 (+http://www.mmi-sc.co.jp/)"
        })
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load crawl cache: {e}")
        return {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save crawl cache: {e}")

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def is_internal(self, url: str) -> bool:
        parsed = urlparse(url)
        # Check if URL belongs to the same domain and is not an asset/file
        is_same_domain = parsed.netloc == "" or parsed.netloc == self.domain
        
        # Exclude common assets and media files
        exclude_exts = [
            ".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", 
            ".pdf", ".zip", ".tar", ".gz", ".xml", ".ico", ".woff", ".ttf"
        ]
        path_lower = parsed.path.lower()
        has_excluded_ext = any(path_lower.endswith(ext) for ext in exclude_exts)
        
        # Exclude mailto and tel links
        is_special_protocol = parsed.scheme in ["mailto", "tel"] or url.startswith("#")
        
        # Exclude external forms or tools (e.g. office forms)
        is_office_form = "forms.office.com" in url
        
        return is_same_domain and not has_excluded_ext and not is_special_protocol and not is_office_form

    def clean_url(self, url: str) -> str:
        # Standardize URL path (strip query parameters and fragments, make absolute)
        absolute_url = urljoin(self.base_url, url)
        parsed = urlparse(absolute_url)
        # Clean path, strip trailing slash if not root, strip fragment/query
        path = parsed.path
        if path.endswith("/") and len(path) > 1:
            path = path[:-1]
        cleaned = f"{parsed.scheme}://{parsed.netloc}{path}"
        return cleaned

    def fetch_dynamic_links(self) -> set:
        """
        Fetches global script files (header.js and footer.js) and extracts internal URLs via regex.
        This ensures we capture all dynamically generated links in menus.
        """
        links = set()
        script_paths = ["common/js/header.js", "common/js/footer.js"]
        
        for sp in script_paths:
            script_url = urljoin(self.base_url, sp)
            try:
                logger.info(f"Fetching JS script for dynamic links: {script_url}")
                r = self.session.get(script_url, timeout=10)
                if r.status_code == 200:
                    content = r.text
                    # Find all href="..." patterns inside javascript document.write or string commands
                    found = re.findall(r'href=["\']([^"\']+)["\']', content)
                    for link in found:
                        if self.is_internal(link):
                            links.add(self.clean_url(link))
            except Exception as e:
                logger.error(f"Error fetching/parsing JS scripts: {e}")
                
        return links

    def crawl(self, force_recrawl: bool = False) -> dict:
        """
        Main crawling routine. Recursively traverses all subpages of the domain.
        Returns a dict mapping URL to page content info.
        """
        to_crawl = {self.clean_url("/")}
        
        # Seed crawling queue with dynamic links found in navigation scripts
        dynamic_links = self.fetch_dynamic_links()
        to_crawl.update(dynamic_links)
        
        crawled = set()
        results = {}
        
        logger.info(f"Starting crawl. Initial queue size: {len(to_crawl)}")

        while to_crawl:
            current_url = to_crawl.pop()
            if current_url in crawled:
                continue
                
            crawled.add(current_url)
            logger.info(f"Crawling URL: {current_url}")

            # Prepare request headers for incremental check if page was crawled before
            headers = {}
            cached_page = self.cache.get(current_url)
            if cached_page and not force_recrawl:
                # Add If-Modified-Since header if last_modified exists in cache
                if "last_modified" in cached_page:
                    headers["If-Modified-Since"] = cached_page["last_modified"]

            try:
                response = self.session.get(current_url, headers=headers, timeout=15)
                
                # Check for HTTP 304 (Not Modified)
                if response.status_code == 304 and cached_page:
                    logger.info(f"Page unmodified (304): {current_url}")
                    results[current_url] = cached_page
                    # Parse existing cached HTML to find new links on page
                    self._extract_links_from_html(cached_page["html"], current_url, to_crawl)
                    continue

                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {current_url}: status {response.status_code}")
                    # If we have cache, fallback to it
                    if cached_page:
                        logger.info(f"Using cache fallback for failed page: {current_url}")
                        results[current_url] = cached_page
                    continue

                html_content = response.text
                content_hash = self._compute_hash(html_content)
                
                # Check for hash changes (if last-modified header is missing or unreliable)
                if cached_page and cached_page.get("hash") == content_hash and not force_recrawl:
                    logger.info(f"Page unmodified (hash match): {current_url}")
                    results[current_url] = cached_page
                    self._extract_links_from_html(html_content, current_url, to_crawl)
                    continue

                # Content has changed or is new
                last_modified = response.headers.get("Last-Modified")
                
                page_data = {
                    "url": current_url,
                    "html": html_content,
                    "hash": content_hash,
                    "last_modified": last_modified,
                    "status_code": response.status_code,
                    "updated": True # Flag to signal rebuild is needed
                }
                
                # Save to results and update cache
                results[current_url] = page_data
                self.cache[current_url] = page_data
                self._save_cache()
                
                # Extract and queue links
                self._extract_links_from_html(html_content, current_url, to_crawl)
                
            except Exception as e:
                logger.error(f"Error crawling {current_url}: {e}")
                # Fallback to cache if available
                if cached_page:
                    results[current_url] = cached_page
                    self._extract_links_from_html(cached_page["html"], current_url, to_crawl)

        return results

    def _extract_links_from_html(self, html_content: str, current_url: str, queue: set):
        """
        Parses page HTML and adds internal URLs to the crawl queue.
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                if self.is_internal(href):
                    cleaned = self.clean_url(href)
                    if cleaned not in queue:
                        queue.add(cleaned)
        except Exception as e:
            logger.error(f"Error extracting links from html of {current_url}: {e}")
