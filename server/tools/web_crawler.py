from crawl4ai import *
from io import StringIO
import json
import os
from mcp.server.fastmcp import FastMCP
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from utils.config import config
from utils.logger import logger

try:
    from .rag.session import get_rag_session

    _rag_available = True
except ImportError:
    _rag_available = False
    logger.debug("RAG module not available")

# crawl4ai drives a headless Chromium via Playwright. The browser binary is a
# separate download that pip / `uv tool install` don't fetch, so the first
# crawl after a fresh install fails with "Executable doesn't exist". We install
# it lazily on that first failure, then retry. Guarded so we try at most once
# per process.
_browser_install_attempted = False


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

        if not url or not url.startswith(("http://", "https://")):
            return json.dumps({"error": True, "message": "Invalid URL"})

        async def _crawl():
            """Run the crawl with stdout muted; returns the crawl result."""
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                async with AsyncWebCrawler(
                    browser_type="none", verbose=False
                ) as crawler:
                    return await crawler.arun(url=url)
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
            if config.get("ENABLE_RAG", False) and _rag_available:
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
                                    "message": "Content ingested into RAG store",
                                    "chunks_indexed": num_chunks,
                                    "metadata": result.metadata,
                                },
                                indent=2,
                            )
                    except Exception as e:
                        logger.exception("RAG ingestion failed: %s", e)

            return json.dumps(
                {
                    "success": True,
                    "url": result.url,
                    "status_code": result.status_code,
                    "content": content,
                    "metadata": result.metadata,
                },
                indent=2,
            )

        except Exception as e:
            logger.error(f"Error during web crawling: {str(e)}", exc_info=True)
            return json.dumps({"error": True, "message": str(e)})
