# graph/state.py — Unified State TypedDict shared by chat and blog subgraphs

from __future__ import annotations
import operator
from typing import TypedDict, List, Optional, Annotated

from langchain_core.messages import BaseMessage


class State(TypedDict, total=False):
    # ---- shared / chat ----
    messages: Annotated[List[BaseMessage], operator.add]
    thread_id: str
    intent: str                     # "chat" | "blog"

    # ---- blog: topic + routing ----
    topic: str
    mode: str                       # closed_book | hybrid | open_book
    needs_research: bool
    queries: List[str]
    as_of: str
    recency_days: int

    # ---- blog: research ----
    evidence: list                  # List[EvidenceItem] — web (Tavily) + pdf-sourced evidence
    pdf_context_used: bool          # True if PDF chunks contributed to evidence

    # ---- blog: plan + sections ----
    plan: Optional[dict]            # Plan object (serialised)
    sections: Annotated[List[tuple], operator.add]

    # ---- blog: reducer / images ----
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str
