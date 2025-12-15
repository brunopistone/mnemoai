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

        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                async with AsyncWebCrawler(
                    browser_type="none", verbose=False
                ) as crawler:
                    result = await crawler.arun(url=url)
            finally:
                sys.stdout = old_stdout

            if not result.success:
                return json.dumps(
                    {"error": True, "message": f"Failed to crawl: {result.status_code}"}
                )

            content = result.markdown

            # If RAG enabled and content is large, ingest into vector DB
            if config.get("ENABLE_RAG", False) and _rag_available:
                from .readers.chunking_helper import __count_tokens as count_tokens

                tokens = count_tokens(content)
                if tokens > config.get("RAG_MAX_TOKENS", 1024 * 8):
                    try:
                        rag = get_rag_session()
                        if rag is not None:
                            num_chunks = rag.ingest(
                                url,
                                content,
                                chunk_size_tokens=int(
                                    config.get("DOC_CHUNK_TOKENS", 2048)
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
            sys.stdout = old_stdout
            logger.error(f"Error during web crawling: {str(e)}", exc_info=True)
            return json.dumps({"error": True, "message": str(e)})
