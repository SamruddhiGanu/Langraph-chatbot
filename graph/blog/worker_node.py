# graph/blog/worker_node.py — Section writer (runs in parallel via fan-out)

import json
from langchain_core.messages import SystemMessage, HumanMessage

from graph.llm import get_llm

WORKER_SYSTEM = """You are a skilled blog writer tasked with writing one section of a blog post.

You will be given:
- The overall blog title
- This section's heading
- Writing guidelines for this section
- Relevant research evidence (with sources)

Guidelines:
- Write in a clear, engaging, professional tone.
- Use markdown formatting (## for H2, **bold**, bullet points as appropriate).
- Do NOT include the blog title itself — only the section heading and content.
- Keep the section focused and on-topic (aim for 150-300 words per section).
- Cite your sources correctly:
  * For evidence with source="web": use inline markdown links [Source Title](URL).
  * For evidence with source="uploaded_pdf": cite as "(from your uploaded document)".
- Do not fabricate facts not supported by the evidence.
"""


def worker_node(state: dict) -> dict:
    """Write a single blog section. Called once per section via fan-out."""
    section_title = state.get("_section_title", "Section")
    guidelines    = state.get("_guidelines", "")
    relevant_urls = state.get("_relevant_urls", [])
    plan_title    = state.get("_plan_title", "")
    evidence_raw  = state.get("evidence") or []
    section_index = state.get("_section_index", 0)

    # Filter evidence to only items relevant to this section
    if relevant_urls:
        relevant_evidence = [
            e for e in evidence_raw
            if e.get("url") in relevant_urls or e.get("source") == "uploaded_pdf"
        ]
    else:
        relevant_evidence = evidence_raw

    evidence_text = json.dumps(relevant_evidence, indent=2)[:3000]

    writer = get_llm("writer", temperature=0.6)
    response = writer.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan_title}\n"
            f"Section heading: {section_title}\n"
            f"Guidelines: {guidelines}\n\n"
            f"Available evidence:\n{evidence_text}"
        )),
    ])

    section_md = f"## {section_title}\n\n{response.content.strip()}"
    return {
        "sections": [(section_index, section_md)]
    }
