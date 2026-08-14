# graph/blog/router_node.py — Decides if blog topic needs web research

from datetime import date

from langchain_core.messages import SystemMessage, HumanMessage

from graph.llm import get_llm
from graph.blog.schemas import RouterDecision

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts that don't change
  (e.g. "what is recursion", "history of the internet").
- hybrid (needs_research=true): mostly evergreen but benefits from current
  examples, recent tools, or up-to-date model names
  (e.g. "best Python web frameworks 2025", "intro to LLMs").
- open_book (needs_research=true): volatile, news-driven, or pricing/policy
  (e.g. "latest AI model releases this week", "current GPT-4 pricing").

If needs_research=true, produce 3-8 high-signal, scoped search queries
that will surface the most useful evidence for a blog writer.
"""


def blog_router_node(state: dict) -> dict:
    """Decide research mode and generate search queries."""
    decider = get_llm("router", temperature=0).with_structured_output(RouterDecision)
    as_of = state.get("as_of") or str(date.today())
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {as_of}"),
    ])
    recency_days = {
        "open_book":   7,
        "hybrid":      45,
        "closed_book": 3650,
    }[decision.mode]
    return {
        "needs_research": decision.needs_research,
        "mode":           decision.mode,
        "queries":        decision.queries or [],
        "recency_days":   recency_days,
        "as_of":          as_of,
    }


def route_next(state: dict) -> str:
    """Conditional edge after blog_router_node."""
    return "research" if state.get("needs_research") else "orchestrator"
