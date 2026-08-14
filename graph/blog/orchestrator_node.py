# graph/blog/orchestrator_node.py — Generates the blog plan and fans out to workers

import json
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send

from graph.llm import get_llm
from graph.blog.schemas import Plan, EvidenceItem

ORCHESTRATOR_SYSTEM = """You are a professional blog post planner.

Given a topic and a pool of research evidence, produce a detailed blog post plan.

Guidelines:
- Create a compelling, specific title (not generic).
- Plan 3-5 body sections (excluding intro and conclusion).
- Each section should have a clear H2 heading and focused writing guidelines.
- Assign relevant evidence URLs to each section so workers can cite them.
- Target a professional, informative tone suitable for a technical audience.
- The blog should be comprehensive but readable (aim for 1000-1500 words total).
"""


def orchestrator_node(state: dict) -> dict:
    """Generate a blog plan from the topic and evidence pool."""
    evidence_raw = state.get("evidence") or []
    evidence_text = json.dumps(evidence_raw, indent=2)[:4000]  # cap context size

    planner = get_llm("writer", temperature=0.4).with_structured_output(Plan)
    plan = planner.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n\n"
            f"Evidence pool:\n{evidence_text}"
        )),
    ])
    return {"plan": plan.model_dump()}


def fanout(state: dict) -> List[Send]:
    """Fan out: send each section task (+ intro + conclusion) to the worker node."""
    plan_dict = state.get("plan") or {}

    sends = []
    # Intro section
    sends.append(Send("worker", {
        **state,
        "_section_title": "Introduction",
        "_section_index": -1,
        "_guidelines": plan_dict.get("intro_guidelines", "Write a compelling introduction."),
        "_relevant_urls": [],
        "_plan_title": plan_dict.get("title", state.get("topic", "")),
    }))

    # Body sections
    for task in (plan_dict.get("tasks") or []):
        sends.append(Send("worker", {
            **state,
            "_section_title": task.get("section_title", "Section"),
            "_section_index": task.get("section_index", 0),
            "_guidelines": task.get("guidelines", ""),
            "_relevant_urls": task.get("relevant_evidence_urls", []),
            "_plan_title": plan_dict.get("title", state.get("topic", "")),
        }))

    # Conclusion section
    sends.append(Send("worker", {
        **state,
        "_section_title": "Conclusion",
        "_section_index": 9999,
        "_guidelines": plan_dict.get("conclusion_guidelines", "Write a strong conclusion."),
        "_relevant_urls": [],
        "_plan_title": plan_dict.get("title", state.get("topic", "")),
    }))

    return sends
