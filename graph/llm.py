# graph/llm.py — Groq client factory

import os
from langchain_groq import ChatGroq


def get_llm(kind: str = "chat", temperature: float = 0.3) -> ChatGroq:
    """Return a Groq LLM instance for the given role.

    kind:
        "router"  — llama-3.3-70b-versatile
        "chat"    — llama-3.3-70b-versatile
        "writer"  — llama-3.3-70b-versatile
    """
    model_name = "llama-3.3-70b-versatile"
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )

