# graph/blog/build_blog_graph.py — Compiles the full blog subgraph

from langgraph.graph import StateGraph, START, END

from graph.state import State
from graph.blog.router_node import blog_router_node, route_next
from graph.blog.research_node import blog_research_node
from graph.blog.orchestrator_node import orchestrator_node, fanout
from graph.blog.worker_node import worker_node
from graph.blog.reducer import merge_content, decide_images, generate_and_place_images

# ---------------------------------------------------------------------------
# Build the blog graph
# ---------------------------------------------------------------------------
_blog_graph = StateGraph(State)

# Nodes
_blog_graph.add_node("router",           blog_router_node)
_blog_graph.add_node("research",         blog_research_node)
_blog_graph.add_node("orchestrator",     orchestrator_node)
_blog_graph.add_node("worker",           worker_node)
_blog_graph.add_node("merge_content",    merge_content)
_blog_graph.add_node("decide_images",    decide_images)
_blog_graph.add_node("generate_images",  generate_and_place_images)

# Edges
_blog_graph.add_edge(START, "router")

# router → research OR orchestrator (skip research for closed_book)
_blog_graph.add_conditional_edges(
    "router",
    route_next,
    {"research": "research", "orchestrator": "orchestrator"},
)

_blog_graph.add_edge("research", "orchestrator")

# orchestrator → fan-out to multiple worker instances
_blog_graph.add_conditional_edges("orchestrator", fanout, ["worker"])

# All workers converge → reducer pipeline
_blog_graph.add_edge("worker", "merge_content")
_blog_graph.add_edge("merge_content", "decide_images")
_blog_graph.add_edge("decide_images", "generate_images")
_blog_graph.add_edge("generate_images", END)

blog_subgraph = _blog_graph.compile()
