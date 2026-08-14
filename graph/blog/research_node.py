# graph/blog/research_node.py — Tavily + PDF retriever merge (implements [C])

import os
import importlib
from datetime import date, timedelta
from typing import List, Optional

# pyrefly: ignore [missing-import]
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage

from graph.llm import get_llm
from graph.blog.schemas import EvidenceItem, EvidencePack


RESEARCH_SYSTEM = """You are a research synthesizer.
Given raw web search results AND excerpts from a user-provided document, produce
a list of EvidenceItem objects.

Rules:
- Only include items where the snippet is non-empty and actually useful.
- For document-derived items, use url="uploaded_pdf" and source="uploaded_pdf".
- Prefer relevant + authoritative sources; do not fabricate content.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; otherwise null.
- Keep snippets concise (≤500 chars). Deduplicate by url+snippet.
- Include at most 15 evidence items total.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tavily_search(query: str, max_results: int = 6) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        tool = TavilySearch(
            max_results=max_results,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
        )
        results = tool.invoke({"query": query})
        # TavilySearch returns a list of dicts or a dict with "results" key
        items = results if isinstance(results, list) else results.get("results", [])
        return [
            {
                "title":        r.get("title", ""),
                "url":          r.get("url", ""),
                "snippet":      r.get("content", r.get("snippet", ""))[:600],
                "published_at": r.get("published_date"),
                "source":       "web",
            }
            for r in (items or [])
            if r.get("url")
        ]
    except Exception as e:
        print(f"[research_node] Tavily error for '{query}': {e}")
        return []


def _pdf_search(query: str, k: int = 4) -> List[dict]:
    """Pull PDF chunks relevant to a research query using the TF-IDF rag_backend."""
    try:
        rag = importlib.import_module("rag_backend")
    except ImportError:
        return []

    if rag.get_loaded_filename() is None:
        return []

    # retrieve_context returns formatted text; we need individual chunks
    # Replicate the internal retrieval logic to get per-chunk results
    import numpy as np
    from collections import Counter

    if rag._tfidf_matrix is None or not rag._chunks:
        return []

    q_tokens = rag._tokenize(query)
    q_counts = Counter(q_tokens)
    q_vec = np.zeros(len(rag._vocab), dtype=np.float32)
    for term, cnt in q_counts.items():
        if term in rag._vocab:
            q_vec[rag._vocab[term]] = cnt
    q_vec = q_vec * rag._idf
    norm = np.linalg.norm(q_vec)
    if norm == 0:
        return []
    q_vec /= norm

    scores = rag._tfidf_matrix @ q_vec
    top_idx = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_idx:
        if scores[idx] < 0.01:
            continue
        page = rag._chunk_meta[idx].get("page", "?")
        results.append({
            "title":        f"Uploaded document — {rag.get_loaded_filename()} (p.{page})",
            "url":          "uploaded_pdf",
            "snippet":      rag._chunks[idx][:600],
            "published_at": None,
            "source":       "uploaded_pdf",
        })
    return results


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(s[:10]) if s else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------
def blog_research_node(state: dict) -> dict:
    queries = (state.get("queries") or [])[:10]

    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q))
        raw.extend(_pdf_search(q))

    pdf_context_used = any(r["source"] == "uploaded_pdf" for r in raw)

    if not raw:
        return {"evidence": [], "pdf_context_used": False}

    # Use LLM to deduplicate, filter, and normalise into EvidencePack
    extractor = get_llm("chat", temperature=0).with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"As-of date: {state.get('as_of', 'unknown')}\n"
            f"Recency requirement: {state.get('recency_days', 45)} days\n\n"
            f"Raw results (JSON list):\n{raw}"
        )),
    ])

    # Dedup: PDF items keyed on title (shared url="uploaded_pdf"), web items keyed on url
    dedup: dict = {}
    for e in pack.evidence:
        key = e.title if e.url == "uploaded_pdf" else e.url
        dedup[key] = e
    evidence = list(dedup.values())

    # Recency filter for open_book (PDF evidence always kept)
    if state.get("mode") == "open_book" and state.get("as_of"):
        try:
            as_of_date = date.fromisoformat(state["as_of"])
            cutoff = as_of_date - timedelta(days=int(state.get("recency_days", 7)))
            evidence = [
                e for e in evidence
                if e.source == "uploaded_pdf"
                or ((d := _iso_to_date(e.published_at)) is not None and d >= cutoff)
            ]
        except Exception:
            pass  # keep all evidence if date parsing fails

    # Serialise to dicts for state storage
    return {
        "evidence": [e.model_dump() for e in evidence],
        "pdf_context_used": pdf_context_used,
    }
