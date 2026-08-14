# graph/chat/agent.py — Chat subgraph: Gemini tool-calling loop (LangGraph v1 compatible)

import json
from typing import Any

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage

from graph.llm import get_llm
from graph.chat.tools import tools
from graph.state import State

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
CHAT_SYSTEM = """You are a helpful AI assistant. You have access to these tools:

- tavily_search: search the live web for current information
- search_pdf: search the user's uploaded PDF document

Guidelines:
- Use tavily_search when the user asks about current events, news, or facts you may not know.
- Use search_pdf when the user asks about the content of their uploaded document.
- If no PDF is uploaded and the user asks about a document, politely ask them to upload one.
- For general knowledge questions, answer directly without using tools.
- Keep responses concise and helpful.
"""

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
_tools_by_name: dict[str, Any] = {t.name: t for t in tools}

# ---------------------------------------------------------------------------
# Lazy-bound LLM
# ---------------------------------------------------------------------------
_llm_with_tools = None


def _get_llm():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm("chat").bind_tools(tools)
    return _llm_with_tools


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def chat_agent_node(state: State) -> dict:
    """Call the LLM with tools bound."""
    msgs = [SystemMessage(content=CHAT_SYSTEM)] + list(state["messages"])
    response = _get_llm().invoke(msgs)
    return {"messages": [response]}


def tool_executor_node(state: State) -> dict:
    """Execute any tool calls requested by the last AI message."""
    last_msg = state["messages"][-1]
    tool_results = []
    for tc in getattr(last_msg, "tool_calls", []):
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id   = tc["id"]
        tool = _tools_by_name.get(tool_name)
        if tool is None:
            result = f"Unknown tool: {tool_name}"
        else:
            try:
                result = tool.invoke(tool_args)
                if not isinstance(result, str):
                    result = json.dumps(result)
            except Exception as e:
                result = f"Tool error: {e}"
        tool_results.append(
            ToolMessage(content=result, tool_call_id=tool_id, name=tool_name)
        )
    return {"messages": tool_results}


def should_continue(state: State) -> str:
    """Return 'tools' if the last message has tool calls, else 'end'."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Build the chat subgraph
# ---------------------------------------------------------------------------
_chat_graph = StateGraph(State)
_chat_graph.add_node("agent", chat_agent_node)
_chat_graph.add_node("tools", tool_executor_node)

_chat_graph.add_edge(START, "agent")
_chat_graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END},
)
_chat_graph.add_edge("tools", "agent")

chat_subgraph = _chat_graph.compile()
