# graph/build_graph.py — Wires intent_router + chat subgraph + blog subgraph into one app

from langgraph.graph import StateGraph, START, END

from graph.state import State
from graph.intent_router import intent_router_node, route_intent
from graph.chat.agent import chat_subgraph
from graph.blog.build_blog_graph import blog_subgraph
from graph.memory import get_checkpointer

_app = None


def build_app():
    """Build and compile the top-level LangGraph application (singleton)."""
    global _app
    if _app is not None:
        return _app

    g = StateGraph(State)

    # Nodes
    g.add_node("intent_router", intent_router_node)
    g.add_node("chat",          chat_subgraph)
    g.add_node("blog",          blog_subgraph)

    # Edges
    g.add_edge(START, "intent_router")
    g.add_conditional_edges(
        "intent_router",
        route_intent,
        {"chat": "chat", "blog": "blog"},
    )
    g.add_edge("chat", END)
    g.add_edge("blog", END)

    _app = g.compile(checkpointer=get_checkpointer())
    return _app
