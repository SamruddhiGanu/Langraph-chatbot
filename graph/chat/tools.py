# graph/chat/tools.py — Chat subgraph tools: Tavily web search + PDF retriever

import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Lazy import of rag_backend so this module can be imported without
# immediately triggering PDF index loading.
# ---------------------------------------------------------------------------
import importlib


def _get_rag():
    """Lazy import of the project's TF-IDF RAG backend."""
    return importlib.import_module("rag_backend")


# ---------------------------------------------------------------------------
# Web search tool
# ---------------------------------------------------------------------------
web_search_tool = TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)


# ---------------------------------------------------------------------------
# PDF retriever tool (wraps existing rag_backend.retrieve_context)
# ---------------------------------------------------------------------------
@tool
def search_pdf(query: str) -> str:
    """Search the user's uploaded PDF document for content relevant to the query.
    Use this tool when the user asks about the content of an uploaded document."""
    rag = _get_rag()
    if rag.get_loaded_filename() is None:
        return "No PDF has been uploaded yet. Please upload a PDF using the sidebar."
    return rag.retrieve_context(query, k=4)


tools = [web_search_tool, search_pdf]
