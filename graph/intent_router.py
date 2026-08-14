# graph/intent_router.py — Top-level graph entry: classify chat vs blog

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

If intent="blog", extract a short, clean topic string (2-8 words).
"""


def intent_router_node(state: State) -> dict:
    """Classify the user's latest message as 'chat' or 'blog'."""
    llm = get_llm("router", temperature=0)
    classifier = llm.with_structured_output(Intent)
    last_user_msg = state["messages"][-1].content
    result = classifier.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=last_user_msg),
    ])
    out: dict = {"intent": result.intent}
    if result.intent == "blog":
        out["topic"] = result.topic or last_user_msg
    return out


def route_intent(state: State) -> str:
    """Conditional edge: dispatch to 'blog' or 'chat' subgraph."""
    return "blog" if state.get("intent") == "blog" else "chat"
