"""Web search tool using Brave Search API."""

import json
import os
from datetime import datetime

from brave_search_python_client import BraveSearch, WebSafeSearchType, WebSearchRequest
from mcp.server.fastmcp import FastMCP

from mnemoai.utils.logger import logger


def register_web_search_tools(mcp: FastMCP) -> None:
    """Register web search tools.

    Args:
        mcp: FastMCP server instance to register tools with
    """

    @mcp.tool()
    async def web_search(
        query: str,
        search_lang: str = "en",
        num_results: int = 10,
        freshness: str = "",
        country: str = "",
        ui_lang: str = "",
    ) -> str:
        """Search the INTERNET for current information, news, and general knowledge.

        For recent or time-sensitive facts, do NOT rely on training knowledge —
        search, and prefer recent results (use `freshness`). Treat any assumption
        about the "latest"/"current" state of the world as stale until a result
        confirms it; use the current calendar year in time-relative queries.

        You MUST ground every factual claim from this tool in its results and
        CITE your sources: include the result title and link for each fact you
        report. Do not answer web questions from memory when results are present.

        Use this for:
        - Current events, news, real-time information
        - General knowledge questions
        - Information NOT from user's local documents
        - Anything requiring up-to-date web content

        DO NOT use this for:
        - Searching user's local documents (use rag_query instead)

        Args:
            query: Search query string
            search_lang: Language to use for the search. The 2 or more character language code for (default: en)
            num_results: Number of results to return (default: 10, max: 20)
            freshness: Optional recency filter. One of pd (past day), pw (past
                week), pm (past month), py (past year), or a date range
                'YYYY-MM-DDtoYYYY-MM-DD'. Empty = no filter.
            country: Optional 2-letter region code (e.g. US, GB) to localize results.
            ui_lang: Optional UI language market code (e.g. en-US). Empty = default.

        Returns:
            JSON string containing search results with titles, links, descriptions, and metadata
        """
        logger.debug(f"Tool web_search called with query: {query}")

        try:
            # Validate inputs
            if not query or not query.strip():
                return json.dumps({"error": True, "message": "Query cannot be empty"})

            # Get Brave API key from environment
            api_key = os.getenv("BRAVE_API_KEY")

            if not api_key:
                return json.dumps(
                    {
                        "error": True,
                        "message": "BRAVE_API_KEY environment variable not set",
                    }
                )

            # Limit num_results to reasonable bounds (Brave API max is 20)
            num_results = max(1, min(num_results, 20))

            # Initialize Brave client
            brave = BraveSearch(api_key=api_key)

            # Create search request. Optional region/recency params are only
            # passed when set — empty strings fail Brave's validation.
            request_kwargs = {
                "q": query.strip(),
                "count": num_results,
                "search_lang": search_lang,
                "safesearch": WebSafeSearchType.off,
            }
            if freshness:
                request_kwargs["freshness"] = freshness
            if country:
                request_kwargs["country"] = country
            if ui_lang:
                request_kwargs["ui_lang"] = ui_lang
            request = WebSearchRequest(**request_kwargs)

            # Execute search
            search_results = await brave.web(request)

            # Process and format results
            formatted_results = {
                "query": query,
                "num_results_requested": num_results,
                "results": [],
            }

            # Extract web results
            if search_results.web and search_results.web.results:
                for i, result in enumerate(search_results.web.results):
                    formatted_result = {
                        "title": result.title or "",
                        "link": result.url or "",
                        "description": result.description or "",
                        "position": i + 1,
                    }

                    # Add published date if available
                    if hasattr(result, "published") and result.published:
                        formatted_result["date"] = result.published

                    formatted_results["results"].append(formatted_result)

            # Add infobox if available (similar to knowledge graph)
            if hasattr(search_results, "infobox") and search_results.infobox:
                infobox = search_results.infobox
                formatted_results["infobox"] = {
                    "title": getattr(infobox, "title", ""),
                    "description": getattr(infobox, "description", ""),
                    "url": getattr(infobox, "url", ""),
                }

            # Add FAQ if available (similar to answer box)
            if hasattr(search_results, "faq") and search_results.faq:
                faq = search_results.faq
                if hasattr(faq, "results") and faq.results:
                    formatted_results["faq"] = []
                    for qa in faq.results[:3]:  # Limit to 3 FAQ items
                        formatted_results["faq"].append(
                            {
                                "question": getattr(qa, "question", ""),
                                "answer": getattr(qa, "answer", ""),
                            }
                        )

            # Add search metadata
            formatted_results["total_results"] = len(formatted_results["results"])
            formatted_results["search_provider"] = "Brave Search"
            formatted_results["current_year"] = datetime.now().year
            if freshness:
                formatted_results["freshness"] = freshness
            if country:
                formatted_results["country"] = country
            if ui_lang:
                formatted_results["ui_lang"] = ui_lang

            return json.dumps(formatted_results, indent=2)

        except Exception as e:
            logger.error(f"Error during Brave web search: {str(e)}", exc_info=True)
            return json.dumps({"error": True, "message": f"Search failed: {str(e)}"})
