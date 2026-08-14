# Implementation Plan — Blog Specialist Agent + Gemini/Tavily Migration

## 0. Scope

This document describes how to extend the existing chatbot (Streamlit UI, LangGraph
orchestration, PDF upload + RAG, web search tool, SQLite-backed memory, LangSmith
tracing, previously running on Groq) to support:

- **[A]** A **Blog Specialist Agent** that researches the web, writes a multi-section
  blog, and generates + places images.
- **[B]** The blog agent internally runs the pipeline
  `router → research → orchestrator → worker (fan-out) → reducer (stitch + images) → END`.
- **[C]** The research step pulls from **both** Tavily web search **and** the user's
  already-uploaded PDF vector store, merging both into one evidence pool.
- **[D]** The system still behaves like a **normal chatbot with tools** (PDF RAG
  retrieval tool, web search tool) for everything that isn't a blog request.
- **[E]** A **provider migration**: Groq → **Google Gemini** (`langchain-google-genai`)
  for text generation and structured output, and **Gemini's native image model**
  (`gemini-2.5-flash-image`) for image generation, with **Tavily** as the search
  provider (already had keys added to `.env`).

The key architectural move: **one top-level LangGraph graph** with an **intent
router** at the entry point that decides whether a turn is a normal chat turn or a
blog-generation request, and dispatches to one of two subgraphs. Both subgraphs share
the same `state`, the same SQLite checkpointer (memory), and the same LangSmith
project, so conversation history and tracing stay unified.

---

## 1. High-Level Architecture

```
                         ┌─────────────────────────────┐
                         │        Streamlit UI          │
                         │  (chat box, PDF uploader,     │
                         │   blog download button)       │
                         └───────────────┬───────────────┘
                                          │ invoke(thread_id, message)
                                          ▼
                         ┌─────────────────────────────┐
                         │   TOP-LEVEL GRAPH (app.py)    │
                         │                                │
                         │   START → intent_router        │
                         │        ├── "chat" ─────────────┼──► CHAT SUBGRAPH
                         │        └── "blog" ─────────────┼──► BLOG SUBGRAPH
                         │                                │
                         │   both subgraphs → END          │
                         └───────────────┬───────────────┘
                                          │
                       ┌──────────────────┴───────────────────┐
                       ▼                                        ▼
         ┌───────────────────────────┐          ┌───────────────────────────────┐
         │       CHAT SUBGRAPH        │          │        BLOG SUBGRAPH           │
         │  agent (Gemini) ⇄ tools    │          │ router → research → orchestr.  │
         │  tools: pdf_retriever,     │          │  → worker(fan-out) → reducer   │
         │         web_search         │          │  reducer = merge → images →    │
         │  loop until no tool calls  │          │            place              │
         └───────────────────────────┘          └───────────────────────────────┘
                       │                                        │
                       └───────────────┬────────────────────────┘
                                        ▼
                         SQLite checkpointer (shared, keyed by thread_id)
                         LangSmith tracing (shared project, tagged by subgraph)
```

**Why one graph instead of two separate apps:** a single `StateGraph` with a
conditional entry edge means memory (checkpointer), tracing, and the PDF vector
store are all wired once. The alternative (two independent LangGraph apps glued
together in Streamlit) duplicates checkpointer setup and makes conversational
continuity (e.g., "now turn that into a blog") harder to support later.

---

## 2. Environment Variables (`.env`)

Remove Groq, keep/add the following:

```env
# LLM
GOOGLE_API_KEY=your_gemini_key

# Search
TAVILY_API_KEY=your_tavily_key

# Tracing (unchanged if already set up)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=chatbot-blog-agent

# Optional: separate model names for cheap routing vs synthesis
GEMINI_ROUTER_MODEL=gemini-2.5-flash-lite
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_WRITER_MODEL=gemini-2.5-pro
GEMINI_IMAGE_MODEL=gemini-2.5-flash-image
```

Install/replace dependencies:

```bash
pip uninstall -y langchain-groq
pip install -U langchain-google-genai google-genai langchain-tavily langgraph langchain-core python-dotenv
```

> Use `langchain-tavily`'s `TavilySearch` (the maintained package) instead of the
> deprecated `langchain_community.tools.tavily_search.TavilySearchResults` used in
> the reference script — same behavior, actively supported.

---

## 3. Project Structure

```
app/
├── app.py                     # Streamlit entrypoint
├── graph/
│   ├── state.py                # Unified State TypedDict
│   ├── intent_router.py        # [entry] chat vs blog classifier
│   ├── llm.py                  # Gemini client factory (chat + writer + router models)
│   ├── memory.py                # SQLite checkpointer setup
│   ├── build_graph.py           # Wires everything into one compiled graph
│   ├── chat/
│   │   ├── agent.py             # Tool-calling chat node
│   │   └── tools.py             # pdf_retriever_tool, web_search_tool
│   └── blog/
│       ├── schemas.py           # Task, Plan, EvidenceItem, ImageSpec, etc.
│       ├── router_node.py       # closed/hybrid/open-book routing
│       ├── research_node.py     # Tavily + PDF retriever merge
│       ├── orchestrator_node.py # Plan generation
│       ├── worker_node.py       # Section writer (fan-out via Send)
│       └── reducer.py           # merge_content → decide_images → generate_and_place_images
├── rag/
│   ├── ingest.py                 # existing PDF ingestion (unchanged)
│   └── store.py                  # existing vector store accessor (unchanged)
└── data/
    ├── chat_memory.sqlite
    └── images/
```

Only `graph/` is new/changed. `rag/` (PDF ingestion + vector store) is assumed to
already exist from the current build and is **reused**, not rebuilt.

---

## 4. Unified State Schema

`graph/state.py`

```python
from __future__ import annotations
import operator
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage

from graph.blog.schemas import Task, Plan, EvidenceItem, ImageSpec


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
    evidence: List[EvidenceItem]     # web (Tavily) + pdf-sourced evidence, unified
    pdf_context_used: bool           # True if PDF chunks contributed to evidence

    # ---- blog: plan + sections ----
    plan: Optional[Plan]
    sections: Annotated[List[tuple], operator.add]

    # ---- blog: reducer / images ----
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str
```

Keeping `messages` in the same state object means the blog subgraph can append its
final markdown as an `AIMessage` back into the conversation, so chat history and
blog output live in one continuous thread.

---

## 5. LLM Provider Migration (Groq → Gemini)

`graph/llm.py`

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(kind: str = "chat", temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    model_map = {
        "router": os.getenv("GEMINI_ROUTER_MODEL", "gemini-2.5-flash-lite"),
        "chat": os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        "writer": os.getenv("GEMINI_WRITER_MODEL", "gemini-2.5-pro"),
    }
    return ChatGoogleGenerativeAI(
        model=model_map[kind],
        temperature=temperature,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
```

**Migration notes / gotchas:**

- Replace every `ChatGroq(model=...)` with `get_llm("chat")` / `get_llm("writer")`.
- `.with_structured_output(PydanticModel)` **is supported** by
  `langchain-google-genai` via function calling, but Gemini is stricter about schema
  shape than OpenAI: avoid deeply nested `Optional[Union[...]]` fields and very long
  enum lists — flatten schemas where possible (the `Task`/`Plan`/`RouterDecision`
  schemas in this project are already flat enough to work as-is).
- Use the **cheap model** (`router`/`flash-lite`) for the intent router and the blog
  router node; use the **stronger model** (`writer`/`pro`) only for the
  orchestrator's plan and the worker's section-writing — this is your token-saving
  story from the earlier discussion (cheap model routes, expensive model writes).
- Image generation is **not** part of `ChatGoogleGenerativeAI` — it uses the
  low-level `google-genai` SDK directly (`gemini-2.5-flash-image`, aka "nano
  banana"), exactly as in the reference script. Keep that call separate (see §8.4).

---

## 6. [D] Intent Router (chat vs. blog) — the graph's entry point

`graph/intent_router.py`

```python
from pydantic import BaseModel
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage
from graph.llm import get_llm
from graph.state import State

class Intent(BaseModel):
    intent: Literal["chat", "blog"]
    topic: str = ""   # populated only when intent == "blog"

INTENT_SYSTEM = """Classify the user's latest message.

Return intent="blog" ONLY if the user is explicitly asking to write, draft,
generate, or create a blog post / article on some topic (e.g. "write a blog about
X", "create an article on Y", "draft a post explaining Z").

Otherwise return intent="chat" — this includes questions, requests to summarize
an uploaded PDF, follow-ups, greetings, or anything not asking for a full blog
post to be authored.

If intent="blog", extract a short, clean topic string.
"""

def intent_router_node(state: State) -> dict:
    llm = get_llm("router", temperature=0)
    classifier = llm.with_structured_output(Intent)
    last_user_msg = state["messages"][-1].content
    result = classifier.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=last_user_msg),
    ])
    out = {"intent": result.intent}
    if result.intent == "blog":
        out["topic"] = result.topic or last_user_msg
    return out

def route_intent(state: State) -> str:
    return "blog" if state["intent"] == "blog" else "chat"
```

This uses the **cheapest model** and a single structured call — negligible latency
and cost, and it's the fork point for `[B]`.

---

## 7. [D] Chat Subgraph (kept, ported to Gemini)

`graph/chat/tools.py` — reuse your existing PDF retriever, swap the search tool:

```python
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from rag.store import get_retriever   # existing function, unchanged

web_search_tool = TavilySearch(max_results=5)

@tool
def pdf_retriever_tool(query: str, thread_id: str) -> str:
    """Search the user's uploaded PDF(s) for content relevant to the query."""
    retriever = get_retriever(thread_id)   # existing per-session vector store
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs) or "No relevant PDF content found."

tools = [web_search_tool, pdf_retriever_tool]
```

`graph/chat/agent.py` — standard LangGraph tool-calling loop, only the model
changed:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from graph.llm import get_llm
from graph.chat.tools import tools
from graph.state import State

CHAT_SYSTEM = """You are a helpful assistant with access to:
- pdf_retriever_tool: search content the user uploaded
- web_search: search the live web
Use tools when the answer depends on the PDF or current information.
"""

llm_with_tools = get_llm("chat").bind_tools(tools)

def chat_agent_node(state: State) -> dict:
    resp = llm_with_tools.invoke([SystemMessage(content=CHAT_SYSTEM)] + state["messages"])
    return {"messages": [resp]}

chat_graph = StateGraph(State)
chat_graph.add_node("agent", chat_agent_node)
chat_graph.add_node("tools", ToolNode(tools))
chat_graph.add_edge(START, "agent")
chat_graph.add_conditional_edges("agent", tools_condition)
chat_graph.add_edge("tools", "agent")
chat_graph.add_edge("agent", END)
chat_subgraph = chat_graph.compile()
```

This is functionally your existing chatbot — only the LLM and the search tool
package changed. This satisfies **[D]**.

---

## 8. [A][B][C] Blog Subgraph

This follows the reference architecture you uploaded almost exactly
(`router → research → orchestrator → worker (fan-out) → reducer`), with two
changes: Gemini instead of OpenAI, and the research node now merges **PDF context**
alongside Tavily results (**[C]**).

### 8.1 Schemas — unchanged

`graph/blog/schemas.py` — copy `Task`, `Plan`, `EvidenceItem`, `RouterDecision`,
`EvidencePack`, `ImageSpec`, `GlobalImagePlan` from the reference script verbatim.
No changes needed; they're provider-agnostic Pydantic models.

### 8.2 Router node — same logic, Gemini model

`graph/blog/router_node.py`

```python
from graph.llm import get_llm
from graph.blog.schemas import RouterDecision
from langchain_core.messages import SystemMessage, HumanMessage

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true, output 3-10 high-signal, scoped queries.
"""

def blog_router_node(state: dict) -> dict:
    decider = get_llm("router", temperature=0).with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
    ])
    recency_days = {"open_book": 7, "hybrid": 45, "closed_book": 3650}[decision.mode]
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: dict) -> str:
    return "research" if state["needs_research"] else "orchestrator"
```

### 8.3 Research node — Tavily + PDF merge (**this is [C]**)

`graph/blog/research_node.py`

```python
import os
from datetime import date, timedelta
from typing import List, Optional
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from graph.llm import get_llm
from graph.blog.schemas import EvidenceItem, EvidencePack
from rag.store import get_retriever

RESEARCH_SYSTEM = """You are a research synthesizer.
Given raw web search results AND excerpts from a user-provided document, produce
EvidenceItem objects.

Rules:
- Only include items with a non-empty url. For document-derived items, use
  url="uploaded_pdf" and source="uploaded_pdf".
- Prefer relevant + authoritative sources; do not fabricate.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null.
- Keep snippets short. Deduplicate by url+snippet.
"""

def _tavily_search(query: str, max_results: int = 6) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})
    items = results.get("results", []) if isinstance(results, dict) else results
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "published_at": r.get("published_date"),
            "source": "web",
        }
        for r in items or []
    ]

def _pdf_search(query: str, thread_id: str, k: int = 4) -> List[dict]:
    """Pull PDF chunks relevant to a research query, tagged as 'uploaded_pdf'."""
    retriever = get_retriever(thread_id)
    if retriever is None:
        return []
    docs = retriever.invoke(query, k=k)
    return [
        {
            "title": f"Uploaded document — {d.metadata.get('source', 'pdf')} (p.{d.metadata.get('page', '?')})",
            "url": "uploaded_pdf",
            "snippet": d.page_content[:600],
            "published_at": None,
            "source": "uploaded_pdf",
        }
        for d in docs
    ]

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(s[:10]) if s else None
    except Exception:
        return None

def blog_research_node(state: dict) -> dict:
    queries = (state.get("queries") or [])[:10]
    thread_id = state.get("thread_id")

    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q))
        if thread_id:
            raw.extend(_pdf_search(q, thread_id))

    pdf_context_used = any(r["source"] == "uploaded_pdf" for r in raw)

    if not raw:
        return {"evidence": [], "pdf_context_used": False}

    extractor = get_llm("chat", temperature=0).with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"As-of date: {state['as_of']}\nRecency days: {state['recency_days']}\n\n"
            f"Raw results:\n{raw}"
        )),
    ])

    # dedup, keeping pdf items keyed on title (since url is shared "uploaded_pdf")
    dedup = {}
    for e in pack.evidence:
        key = e.url if e.url != "uploaded_pdf" else e.title
        dedup[key] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        # keep all uploaded_pdf evidence (undated, user-provided) + recency-filtered web evidence
        evidence = [
            e for e in evidence
            if e.source == "uploaded_pdf"
            or ((d := _iso_to_date(e.published_at)) and d >= cutoff)
        ]

    return {"evidence": evidence, "pdf_context_used": pdf_context_used}
```

**Design notes for [C]:**
- PDF search runs **per query**, using the same queries the router generated for
  web search, so PDF and web evidence are topically aligned rather than a generic
  whole-document dump.
- PDF-sourced `EvidenceItem`s are tagged `source="uploaded_pdf"` so the worker
  prompt (below) can cite them distinctly (e.g. `[Source: your uploaded document]`
  instead of a URL link).
- If the user hasn't uploaded a PDF, `get_retriever` returns `None` and this
  degrades gracefully to web-only research — no special-casing needed elsewhere.
- Recency filtering (for `open_book` mode) only applies to web evidence; PDF
  content has no publish date and is assumed relevant regardless of "as of" cutoff.

### 8.4 Orchestrator, Worker — same logic as reference, Gemini model swapped in

`graph/blog/orchestrator_node.py` and `graph/blog/worker_node.py`: copy the
reference implementation's `orchestrator_node`, `fanout`, and `worker_node`
functions verbatim, replacing `llm = ChatOpenAI(model="gpt-4.1-mini")` with
`llm = get_llm("writer")`. The `Send(...)` fan-out mechanism for parallel section
writing is provider-agnostic and needs no change.

One addition to `WORKER_SYSTEM` to handle PDF-sourced evidence cleanly:

```python
WORKER_SYSTEM = """... (same as reference) ...

Citation format:
- For evidence with source="web": cite as [Source](URL).
- For evidence with source="uploaded_pdf": cite as "(from your uploaded document)"
  instead of a URL link.
"""
```

### 8.5 Reducer — merge, decide images, generate images (Gemini image model)

`graph/blog/reducer.py`: copy `merge_content` and `decide_images` from the
reference script unchanged (model swapped to `get_llm("writer")`). For image
generation, keep `_gemini_generate_image_bytes` exactly as in the reference — it
already uses `gemini-2.5-flash-image` via the `google-genai` SDK, which is the
correct approach; just point `api_key` at `os.environ["GOOGLE_API_KEY"]` (same var
now used for both text and image, since both are Gemini).

Compile the three-node reducer as its own subgraph exactly as in the reference
(`merge_content → decide_images → generate_and_place_images`), then compose the
full blog graph:

```python
from langgraph.graph import StateGraph, START, END
from graph.state import State
from graph.blog.router_node import blog_router_node, route_next
from graph.blog.research_node import blog_research_node
from graph.blog.orchestrator_node import orchestrator_node, fanout
from graph.blog.worker_node import worker_node
from graph.blog.reducer import reducer_subgraph

blog_graph = StateGraph(State)
blog_graph.add_node("router", blog_router_node)
blog_graph.add_node("research", blog_research_node)
blog_graph.add_node("orchestrator", orchestrator_node)
blog_graph.add_node("worker", worker_node)
blog_graph.add_node("reducer", reducer_subgraph)

blog_graph.add_edge(START, "router")
blog_graph.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
blog_graph.add_edge("research", "orchestrator")
blog_graph.add_conditional_edges("orchestrator", fanout, ["worker"])
blog_graph.add_edge("worker", "reducer")
blog_graph.add_edge("reducer", END)

blog_subgraph = blog_graph.compile()
```

Finally, append the blog's `final` markdown back into the shared conversation so
the chat UI and history stay unified:

```python
from langchain_core.messages import AIMessage

def blog_to_message_node(state: State) -> dict:
    return {"messages": [AIMessage(content=state["final"])]}
```

Add this as a thin node after `reducer` (or fold it into `generate_and_place_images`'s
return) so the blog subgraph's output lands in `state["messages"]` just like a
normal chat response.

---

## 9. Wiring the Top-Level Graph

`graph/build_graph.py`

```python
from langgraph.graph import StateGraph, START, END
from graph.state import State
from graph.intent_router import intent_router_node, route_intent
from graph.chat.agent import chat_subgraph
from graph.blog.build_blog_graph import blog_subgraph   # from §8.5, incl. blog_to_message_node
from graph.memory import get_checkpointer

def build_app():
    g = StateGraph(State)
    g.add_node("intent_router", intent_router_node)
    g.add_node("chat", chat_subgraph)
    g.add_node("blog", blog_subgraph)

    g.add_edge(START, "intent_router")
    g.add_conditional_edges("intent_router", route_intent, {"chat": "chat", "blog": "blog"})
    g.add_edge("chat", END)
    g.add_edge("blog", END)

    return g.compile(checkpointer=get_checkpointer())
```

`graph/memory.py` — unchanged from your current SQLite setup:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

def get_checkpointer():
    conn = sqlite3.connect("data/chat_memory.sqlite", check_same_thread=False)
    return SqliteSaver(conn)
```

Because both subgraphs are nodes inside the *same* compiled graph with *one*
checkpointer, `thread_id`-scoped memory now covers chat turns and blog-generation
turns identically — no separate persistence layer needed.

---

## 10. Streamlit Frontend Changes

`app.py` — key additions on top of your existing chat loop:

```python
import streamlit as st
from graph.build_graph import build_app

app = build_app()

# ... existing PDF uploader logic feeding rag/ingest.py stays as-is ...

if prompt := st.chat_input("Ask a question or say 'write a blog about ...'"):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner("Thinking..."):
        result = app.invoke(
            {"messages": [HumanMessage(content=prompt)], "thread_id": st.session_state.thread_id,
             "as_of": str(date.today())},
            config=config,
        )
    last = result["messages"][-1]
    st.chat_message("assistant").markdown(last.content)

    # If a blog was just generated, surface the images + a download button
    if result.get("final"):
        for img in Path("images").glob("*.png"):
            st.image(str(img))
        st.download_button("Download blog.md", data=result["final"], file_name="blog.md")
```

Notes:
- Blog generation is multi-step and slower than a normal chat turn — use
  `st.status("Researching...", expanded=True)` with manual state updates
  (`st.status.update(label="Writing sections...")`) around the `app.invoke` call if
  you want visible progress instead of a single spinner; LangGraph's `.stream()`
  method (instead of `.invoke()`) lets you push per-node updates to the UI in
  real time if you want this polish later.
- `thread_id` should be a stable per-session UUID (`st.session_state.thread_id`,
  set once via `uuid.uuid4()`), used both for the checkpointer and for
  `get_retriever(thread_id)` so each user's uploaded PDFs stay session-scoped.

---

## 11. LangSmith Tracing

No structural change needed — `LANGCHAIN_TRACING_V2=true` traces the whole
compiled graph automatically, including subgraph nodes. Two small improvements
worth adding:

- Tag runs so chat vs. blog traces are filterable in the LangSmith UI:

```python
result = app.invoke(inputs, config={
    "configurable": {"thread_id": thread_id},
    "tags": ["blog"] if looks_like_blog_request else ["chat"],
    "metadata": {"pdf_uploaded": bool(get_retriever(thread_id))},
})
```

(The `tags` can also just be added post-hoc based on `result["intent"]`, since the
intent is only known after the first node runs — tag reactively via
`LANGCHAIN_PROJECT`-level filtering on `intent` if you'd rather not guess up front.)

- Because worker nodes fan out via `Send`, each parallel section-writing call shows
  as its own child run in LangSmith — useful for verifying the parallelism is
  actually happening and for spotting slow/expensive sections.

---

## 12. Testing Plan

1. **Intent routing accuracy** — hand-write ~20 messages (10 clear blog requests,
   10 clear chat/PDF questions) and assert `route_intent` returns the right branch.
2. **PDF-merge correctness** — upload a PDF with distinctive facts, request a blog
   on a *hybrid*-mode topic that should surface PDF evidence; assert
   `pdf_context_used=True` and that at least one section cites
   "(from your uploaded document)".
3. **No-PDF fallback** — same blog request with no PDF uploaded in the session;
   assert the graph completes without error (`get_retriever` returns `None`
   cleanly).
4. **Structured output stability** — run the router/orchestrator/reducer's
   structured-output calls 10x on the same input and check for schema-validation
   failures (Gemini function-calling structured output is generally solid but
   worth a smoke test after the OpenAI→Gemini swap).
5. **Image generation failure path** — temporarily break `GOOGLE_API_KEY` for the
   image call only and confirm `generate_and_place_images` falls back to the
   `[IMAGE GENERATION FAILED]` markdown block instead of crashing the whole run.
6. **Regression on existing chat behavior** — re-run your existing chatbot test
   cases (PDF Q&A, web search Q&A) against the new `chat` subgraph to confirm the
   Groq→Gemini swap didn't change tool-calling behavior.

---

## 13. Rollout Order (suggested)

1. Swap chat subgraph's LLM Groq → Gemini; swap search tool to `langchain-tavily`.
   Ship and confirm the existing chatbot still works end-to-end (**[D], [E]**
   partially done).
2. Add the blog schemas + router/research/orchestrator/worker/reducer nodes as a
   **standalone** compiled graph, tested in isolation (e.g. a small script that
   calls `blog_subgraph.invoke(...)` directly) before wiring it into the main app
   (**[A], [B]**).
3. Add PDF merge into the research node (**[C]**) — test with and without an
   uploaded PDF.
4. Add the intent router and wire both subgraphs into one top-level graph with the
   shared checkpointer (**[D]+[A] unified**).
5. Update Streamlit to handle the blog-output rendering path (images, download
   button, progress messaging).
6. Add LangSmith tags and re-verify tracing coverage across both paths.

---

## 14. Known Limitations / Follow-ups

- **Cost**: each blog generation makes ~1 (router) + N (research, per query) +
  1 (evidence extraction) + 1 (orchestrator) + N (workers, parallel) + 2 (reducer)
  + up to 3 (image gen) model calls — noticeably more expensive than a chat turn.
  Worth surfacing an estimated cost/time to the user before running (e.g. "this
  will take ~60-90s and generate up to 3 images").
- **Gemini structured output edge cases**: if you hit schema-validation errors on
  complex nested models, simplify (e.g. flatten `GlobalImagePlan` into two separate
  structured calls) rather than fighting the schema.
- **Intent misclassification**: ambiguous prompts like "can you summarize this into
  a blog-style post?" may misroute; consider adding a confirmation step
  ("It looks like you want a full blog post on X — proceed?") before triggering the
  expensive blog pipeline, especially once you add the security/rate-limiting work
  discussed earlier for cost-abuse protection.
- **PDF evidence has no recency signal**: for `open_book` blog requests, PDF
  content is currently always included regardless of the recency filter applied to
  web evidence — acceptable for now, but worth revisiting if you want strict
  "as-of" correctness later.
