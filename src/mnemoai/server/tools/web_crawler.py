import json
import sys
from io import StringIO

from crawl4ai import *
from mcp.server.fastmcp import FastMCP

from mnemoai.utils.config import config
from mnemoai.utils.logger import logger

from .safety import classify_url


def _rag_session():
    """Return ``get_rag_session`` if the RAG extra is installed, else None.

    Resolved per call rather than probed at import time, so importing this tool
    doesn't drag in ``.rag`` → faiss (and its OpenMP runtime) when RAG is off.
    """
    try:
        from .rag.session import get_rag_session
    except ImportError:
        logger.debug("RAG module not available")
        return None
    return get_rag_session

# crawl4ai drives a headless Chromium via Playwright. The browser binary is a
# separate download that pip / `uv tool install` don't fetch, so the first
# crawl after a fresh install fails with "Executable doesn't exist". We install
# it lazily on that first failure, then retry. Guarded so we try at most once
# per process.
_browser_install_attempted = False

# Fallback cap on inline-returned markdown (chars) when the RAG-offload path
# isn't taken (RAG disabled/unavailable, or page under the RAG token threshold).
# The RAG path stays the primary large-page handler; this only bounds context.
_MAX_INLINE_CHARS = 100_000


def _is_missing_browser_error(exc: Exception) -> bool:
    """True if the exception is Playwright's missing-browser launch error."""
    msg = str(exc).lower()
    return "executable doesn't exist" in msg or "playwright install" in msg


def _install_playwright_chromium() -> bool:
    """Download the Playwright Chromium build into the current environment.

    Returns True on success. Uses the running interpreter so it lands in the
    same (possibly isolated `uv tool`) environment as the server.
    """
    import subprocess

    logger.info("Installing Playwright Chromium (one-time, ~260MB)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            capture_output=True,
        )
        logger.info("Playwright Chromium installed.")
        return True
    except Exception as e:
        logger.error(f"Failed to install Playwright Chromium: {e}")
        return False


def register_web_crawler_tools(mcp: FastMCP) -> None:
    """Register web search tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    async def web_crawler(url: str) -> str:
        """Crawl a web page to extract its content.
        This tool fetches the content of a given URL and extracts the main text, metadata, and links. The tool MUST be used when the user requests information from a specific web page.

        Use this for:
        - Extracting content from a specific web page
        - Analyzing the structure and metadata of a web page

        Args:
            url: URL of the web page to crawl

        Returns:
            JSON string containing page content, and metadata
        """
        logger.debug(f"Tool web_crawler called with url: {url}")

        # Scheme + destination check. The fetched page becomes model input, so an
        # internal address (localhost, RFC1918, 169.254.169.254) must not be
        # reachable from a prompt. Resolves DNS before fetching.
        url_verdict = classify_url(url)
        if url_verdict.blocked:
            logger.warning("web_crawler blocked url=%s: %s", url, url_verdict.reason)
            return json.dumps(
                {"error": True, "blocked": True, "message": url_verdict.reason}
            )

        # Explicit per-page crawl timeout (ms); tunable via config, code default
        # so no config edit is required to reach existing installs. Guarded so a
        # malformed WEB_CRAWL (bare key -> None, or a non-numeric value) falls
        # back to the default instead of raising outside the error-handling try.
        try:
            web_crawl_cfg = config.get("WEB_CRAWL", {}) or {}
            page_timeout_ms = int(web_crawl_cfg.get("PAGE_TIMEOUT_MS", 60000))
        except (AttributeError, TypeError, ValueError):
            page_timeout_ms = 60000
        run_config = CrawlerRunConfig(page_timeout=page_timeout_ms)

        async def _crawl():
            """Run the crawl with stdout muted; returns the crawl result."""
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                async with AsyncWebCrawler(
                    browser_type="none", verbose=False
                ) as crawler:
                    return await crawler.arun(url=url, config=run_config)
            finally:
                sys.stdout = old_stdout

        try:
            global _browser_install_attempted
            try:
                result = await _crawl()
            except Exception as e:
                # First crawl after a fresh install: the Chromium binary is
                # missing. Install it once, then retry.
                if _is_missing_browser_error(e) and not _browser_install_attempted:
                    _browser_install_attempted = True
                    if _install_playwright_chromium():
                        result = await _crawl()
                    else:
                        return json.dumps({
                            "error": True,
                            "message": "Web crawling needs the Playwright browser. "
                            "Run: python -m playwright install chromium",
                        })
                else:
                    raise

            if not result.success:
                return json.dumps(
                    {"error": True, "message": f"Failed to crawl: {result.status_code}"}
                )

            content = result.markdown

            # If RAG enabled and content is large, ingest into vector DB
            get_rag_session = (
                _rag_session() if config.get("ENABLE_RAG", False) else None
            )
            if get_rag_session is not None:
                from .readers.chunking_helper import __count_tokens as count_tokens

                tokens = count_tokens(content)
                if tokens > config.get("RAG", {}).get("MAX_TOKENS", 1024 * 8):
                    try:
                        rag = get_rag_session()
                        if rag is not None:
                            num_chunks = rag.ingest(
                                url,
                                content,
                                chunk_size_tokens=int(
                                    config.get("RAG", {}).get("CHUNK_TOKENS", 1024)
                                ),
                            )

                            return json.dumps(
                                {
                                    "success": True,
                                    "url": result.url,
                                    "message": (
                                        f"The page was large ({tokens} tokens), so "
                                        f"it was indexed into the document store "
                                        f"({num_chunks} chunks) instead of returned "
                                        f"inline. Retrieve the parts you need with "
                                        f"search_in_documents(query=...) — do NOT "
                                        f"answer from memory; the page content is "
                                        f"only available via that search."
                                    ),
                                    "chunks_indexed": num_chunks,
                                    "next_step": "search_in_documents",
                                    "metadata": result.metadata,
                                },
                                indent=2,
                            )
                    except Exception as e:
                        logger.exception("RAG ingestion failed: %s", e)

            # Fallback inline return (RAG offload not taken). Cap oversized
            # markdown so a huge page can't blow up the model context. The RAG
            # path above is the primary large-page handler and is untouched.
            truncated = False
            if content and len(content) > _MAX_INLINE_CHARS:
                content = (
                    content[:_MAX_INLINE_CHARS]
                    + f"\n\n[... content truncated at {_MAX_INLINE_CHARS} chars. "
                    + "Enable RAG or re-crawl to index this page and use "
                    + "search_in_documents to retrieve the rest — do not assume "
                    + "the omitted content.]"
                )
                truncated = True

            return json.dumps(
                {
                    "success": True,
                    "url": result.url,
                    "status_code": result.status_code,
                    "content": content,
                    "truncated": truncated,
                    "metadata": result.metadata,
                },
                indent=2,
            )

        except Exception as e:
            logger.error(f"Error during web crawling: {str(e)}", exc_info=True)
            return json.dumps({"error": True, "message": str(e)})
